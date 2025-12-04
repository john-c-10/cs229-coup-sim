import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch

from .config import (
    SEQUENCE_LENGTH,
    BURN_IN_LENGTH,
    TRAIN_LENGTH,
    USE_PRIORITIZED_REPLAY,
    PER_ALPHA,
    PER_BETA_START,
    PER_BETA_END,
    PER_BETA_FRAMES,
    PER_EPSILON,
    REPLAY_BUFFER_CAPACITY,
    GAMMA,
    N_STEP,
    CLIP_REWARDS,
    REWARD_CLIP_MIN,
    REWARD_CLIP_MAX,
    DEVICE,
)

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    legal_actions_mask: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SequenceData:
    observations: np.ndarray # (seq_len, obs_dim)
    actions: np.ndarray # (seq_len,)
    rewards: np.ndarray # (seq_len,)
    next_observations: np.ndarray # (seq_len, obs_dim)
    dones: np.ndarray # (seq_len,)
    legal_masks: np.ndarray # (seq_len, num_actions)
    init_hidden_h: Optional[np.ndarray] = None # (num_layers, hidden_dim)
    init_hidden_c: Optional[np.ndarray] = None # (num_layers, hidden_dim) for LSTM
    n_step_returns: Optional[np.ndarray] = None # (train_len,)
    n_step_next_obs: Optional[np.ndarray] = None # (train_len, obs_dim)
    n_step_dones: Optional[np.ndarray] = None # (train_len,)

class SumTree:
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.num_entries = 0
    
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    @property
    def total(self) -> float:
        return self.tree[0]
    
    @property
    def max_priority(self) -> float:
        return max(self.tree[self.capacity - 1:self.capacity - 1 + self.num_entries])
    
    @property
    def min_priority(self) -> float:
        if self.num_entries == 0:
            return 0.0
        priorities = self.tree[self.capacity - 1:self.capacity - 1 + self.num_entries]
        return min(p for p in priorities if p > 0)
    
    def add(self, priority: float, data: Any):
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.num_entries = min(self.num_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class EpisodeBuffer:
    
    def __init__(self, n_step: int = N_STEP, gamma: float = GAMMA):
        self.n_step = n_step
        self.gamma = gamma
        self.transitions: List[Transition] = []
        self.hidden_states: List[Tuple[np.ndarray, ...]] = []
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        legal_mask: np.ndarray,
        info: Dict[str, Any],
        hidden_state: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            legal_actions_mask=legal_mask,
            info=info,
        )
        self.transitions.append(transition)
        
        if hidden_state is not None:
            h = hidden_state[0].cpu().numpy() if isinstance(hidden_state[0], torch.Tensor) else hidden_state[0]
            if len(hidden_state) > 1:
                c = hidden_state[1].cpu().numpy() if isinstance(hidden_state[1], torch.Tensor) else hidden_state[1]
                self.hidden_states.append((h, c))
            else:
                self.hidden_states.append((h, None))
    
    def _clip_reward(self, reward: float) -> float:
        if CLIP_REWARDS:
            return np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
        return reward
    
    def _compute_n_step_return(self, start_idx: int) -> Tuple[float, np.ndarray, bool]:
        n_step_return = 0.0
        gamma_power = 1.0
        
        for i in range(self.n_step):
            idx = start_idx + i
            if idx >= len(self.transitions):
                break
            
            trans = self.transitions[idx]
            n_step_return += gamma_power * self._clip_reward(trans.reward)
            gamma_power *= self.gamma
            
            if trans.done:
                return n_step_return, trans.next_obs, True
        
        end_idx = min(start_idx + self.n_step, len(self.transitions)) - 1
        return n_step_return, self.transitions[end_idx].next_obs, False
    
    def create_sequences(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        burn_in_length: int = BURN_IN_LENGTH,
    ) -> List[SequenceData]:
        if len(self.transitions) < sequence_length:
            return self._create_padded_sequence(sequence_length, burn_in_length)
        
        sequences = []
        train_length = sequence_length - burn_in_length
        
        for start_idx in range(0, len(self.transitions) - sequence_length + 1, train_length):
            seq = self._create_sequence_at(start_idx, sequence_length, burn_in_length)
            sequences.append(seq)
        
        if len(self.transitions) >= sequence_length:
            last_start = len(self.transitions) - sequence_length
            if last_start > 0 and last_start % train_length != 0:
                seq = self._create_sequence_at(last_start, sequence_length, burn_in_length)
                sequences.append(seq)
        
        return sequences
    
    def _create_sequence_at(
        self,
        start_idx: int,
        sequence_length: int,
        burn_in_length: int,
    ) -> SequenceData:
        train_length = sequence_length - burn_in_length
        
        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        done_list = []
        legal_mask_list = []
        
        for i in range(sequence_length):
            trans = self.transitions[start_idx + i]
            obs_list.append(trans.obs)
            action_list.append(trans.action)
            reward_list.append(self._clip_reward(trans.reward))
            next_obs_list.append(trans.next_obs)
            done_list.append(trans.done)
            legal_mask_list.append(trans.legal_actions_mask)
        
        n_step_returns = []
        n_step_next_obs = []
        n_step_dones = []
        
        for i in range(burn_in_length, sequence_length):
            ret, next_obs, done = self._compute_n_step_return(start_idx + i)
            n_step_returns.append(ret)
            n_step_next_obs.append(next_obs)
            n_step_dones.append(done)
        
        init_h, init_c = None, None
        if self.hidden_states and start_idx < len(self.hidden_states):
            init_h, init_c = self.hidden_states[start_idx]
        
        return SequenceData(
            observations=np.stack(obs_list),
            actions=np.array(action_list),
            rewards=np.array(reward_list),
            next_observations=np.stack(next_obs_list),
            dones=np.array(done_list),
            legal_masks=np.stack(legal_mask_list),
            init_hidden_h=init_h,
            init_hidden_c=init_c,
            n_step_returns=np.array(n_step_returns),
            n_step_next_obs=np.stack(n_step_next_obs),
            n_step_dones=np.array(n_step_dones),
        )
    
    def _create_padded_sequence(
        self,
        sequence_length: int,
        burn_in_length: int,
    ) -> List[SequenceData]:
        if len(self.transitions) == 0:
            return []
        
        num_pad = sequence_length - len(self.transitions)
        
        pad_obs = np.zeros_like(self.transitions[0].obs)
        pad_mask = np.zeros_like(self.transitions[0].legal_actions_mask)
        
        obs_list = [pad_obs] * num_pad + [t.obs for t in self.transitions]
        action_list = [0] * num_pad + [t.action for t in self.transitions]
        reward_list = [0.0] * num_pad + [self._clip_reward(t.reward) for t in self.transitions]
        next_obs_list = [pad_obs] * num_pad + [t.next_obs for t in self.transitions]
        done_list = [False] * num_pad + [t.done for t in self.transitions]
        legal_mask_list = [pad_mask] * num_pad + [t.legal_actions_mask for t in self.transitions]
        
        train_length = sequence_length - burn_in_length
        n_step_returns = []
        n_step_next_obs = []
        n_step_dones = []
        
        for i in range(train_length):
            actual_idx = burn_in_length + i - num_pad
            if actual_idx < 0:
                n_step_returns.append(0.0)
                n_step_next_obs.append(pad_obs)
                n_step_dones.append(False)
            else:
                ret, next_obs, done = self._compute_n_step_return(actual_idx)
                n_step_returns.append(ret)
                n_step_next_obs.append(next_obs)
                n_step_dones.append(done)
        
        return [SequenceData(
            observations=np.stack(obs_list),
            actions=np.array(action_list),
            rewards=np.array(reward_list),
            next_observations=np.stack(next_obs_list),
            dones=np.array(done_list),
            legal_masks=np.stack(legal_mask_list),
            init_hidden_h=None,
            init_hidden_c=None,
            n_step_returns=np.array(n_step_returns),
            n_step_next_obs=np.stack(n_step_next_obs),
            n_step_dones=np.array(n_step_dones),
        )]
    
    def clear(self):
        self.transitions.clear()
        self.hidden_states.clear()

class PrioritizedSequenceReplayBuffer:
    
    def __init__(
        self,
        capacity: int = REPLAY_BUFFER_CAPACITY,
        alpha: float = PER_ALPHA,
        beta_start: float = PER_BETA_START,
        beta_end: float = PER_BETA_END,
        beta_frames: int = PER_BETA_FRAMES,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        
        self.tree = SumTree(capacity)
        self.frame_count = 0
        self.max_priority = 1.0
    
    @property
    def beta(self) -> float:
        fraction = min(1.0, self.frame_count / self.beta_frames)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def add(self, sequence: SequenceData, priority: Optional[float] = None):
        if priority is None:
            priority = self.max_priority
        
        priority = (priority + PER_EPSILON) ** self.alpha
        self.tree.add(priority, sequence)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[List[SequenceData], np.ndarray, np.ndarray]:
        sequences = []
        indices = []
        priorities = []
        
        segment = self.tree.total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            sequences.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        min_priority = self.tree.min_priority
        if min_priority == 0:
            min_priority = PER_EPSILON
        
        priorities = np.array(priorities)
        max_weight = (min_priority / self.tree.total * self.tree.num_entries) ** (-self.beta)
        
        weights = (priorities / self.tree.total * self.tree.num_entries) ** (-self.beta)
        weights = weights / max_weight
        
        self.frame_count += batch_size
        
        return sequences, np.array(indices), weights.astype(np.float32)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            priority = (abs(priority) + PER_EPSILON) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.num_entries


class UniformSequenceReplayBuffer:
    
    def __init__(self, capacity: int = REPLAY_BUFFER_CAPACITY):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, sequence: SequenceData, priority: Optional[float] = None):
        self.buffer.append(sequence)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[List[SequenceData], np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sequences = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return sequences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        pass
    
    def __len__(self) -> int:
        return len(self.buffer)

def create_replay_buffer(
    capacity: int = REPLAY_BUFFER_CAPACITY,
    use_prioritized: bool = USE_PRIORITIZED_REPLAY,
):
    if use_prioritized:
        return PrioritizedSequenceReplayBuffer(capacity=capacity)
    else:
        return UniformSequenceReplayBuffer(capacity=capacity)

def collate_sequences(
    sequences: List[SequenceData],
) -> Dict[str, torch.Tensor]:
    batch = {
        "observations": torch.tensor(
            np.stack([s.observations for s in sequences]),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "actions": torch.tensor(
            np.stack([s.actions for s in sequences]),
            dtype=torch.long,
            device=DEVICE,
        ),
        "rewards": torch.tensor(
            np.stack([s.rewards for s in sequences]),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "next_observations": torch.tensor(
            np.stack([s.next_observations for s in sequences]),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "dones": torch.tensor(
            np.stack([s.dones for s in sequences]),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "legal_masks": torch.tensor(
            np.stack([s.legal_masks for s in sequences]),
            dtype=torch.bool,
            device=DEVICE,
        ),
        "n_step_returns": torch.tensor(
            np.stack([s.n_step_returns for s in sequences]),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "n_step_next_obs": torch.tensor(
            np.stack([s.n_step_next_obs for s in sequences]),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "n_step_dones": torch.tensor(
            np.stack([s.n_step_dones for s in sequences]),
            dtype=torch.float32,
            device=DEVICE,
        ),
    }
    
    if sequences[0].init_hidden_h is not None:
        h_list = [s.init_hidden_h for s in sequences]
        batch["init_hidden_h"] = torch.tensor(
            np.stack(h_list, axis=1),
            dtype=torch.float32,
            device=DEVICE,
        )
        
        if sequences[0].init_hidden_c is not None:
            c_list = [s.init_hidden_c for s in sequences]
            batch["init_hidden_c"] = torch.tensor(
                np.stack(c_list, axis=1),
                dtype=torch.float32,
                device=DEVICE,
            )
        else:
            batch["init_hidden_c"] = None
    else:
        batch["init_hidden_h"] = None
        batch["init_hidden_c"] = None
    
    return batch
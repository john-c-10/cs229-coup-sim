import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional, Dict, Any, List
import copy
import os


#agent action selection and learning details


from .config import (
    DEVICE,
    NUM_ACTIONS,
    OBS_DIM,
    GAMMA,
    N_STEP,
    LEARNING_RATE,
    ADAM_EPS,
    BATCH_SIZE,
    TARGET_UPDATE_FREQ,
    USE_POLYAK,
    POLYAK_TAU,
    GRAD_CLIP_NORM,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_FRAMES,
    USE_NOISY_NETS,
    USE_DISTRIBUTIONAL,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    BURN_IN_LENGTH,
    TRAIN_LENGTH,
    SEQUENCE_LENGTH,
    CHECKPOINT_DIR,
)
from .networks import DuelingDRQN, create_network
from .replay import (
    PrioritizedSequenceReplayBuffer,
    EpisodeBuffer,
    SequenceData,
    collate_sequences,
    create_replay_buffer,
)


class DQNAgent:
    
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        gamma: float = GAMMA,
        n_step: int = N_STEP,
        learning_rate: float = LEARNING_RATE,
        buffer_capacity: int = 50_000,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_n = gamma ** n_step
        
        # double DQN: online for action selection, target for value eval
        self.online_net = create_network(obs_dim, num_actions)
        self.target_net = create_network(obs_dim, num_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate,
            eps=ADAM_EPS,
        )
        
        self.replay_buffer = create_replay_buffer(capacity=buffer_capacity)
        
        self.episode_buffer = EpisodeBuffer(n_step=n_step, gamma=gamma)
        
        self.hidden_state: Optional[Tuple[torch.Tensor, ...]] = None
        
        self.total_steps = 0
        self.training_steps = 0
        
        self.epsilon = EPSILON_START
        
        if USE_DISTRIBUTIONAL:
            self.support = torch.linspace(V_MIN, V_MAX, NUM_ATOMS, device=DEVICE)
            self.delta_z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)
    
    @property
    def current_epsilon(self) -> float:
        if USE_NOISY_NETS:
            return 0.0
        return self.epsilon
    
    def _update_epsilon(self):
        self.epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * self.total_steps / EPSILON_DECAY_FRAMES
        )
    
    def select_action(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        training: bool = True,
    ) -> Tuple[int, Tuple[torch.Tensor, ...]]:
        self.total_steps += 1
        self._update_epsilon()
        
        if USE_NOISY_NETS and training:
            self.online_net.reset_noise()
        
        # eps-greedy: random among legal actions
        if training and not USE_NOISY_NETS and np.random.random() < self.epsilon:
            legal_actions = np.where(legal_mask)[0]
            action = np.random.choice(legal_actions)
            
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                mask_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
                _, self.hidden_state = self.online_net(obs_tensor, self.hidden_state, mask_tensor)
            
            return action, self.hidden_state
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
            
            q_values, self.hidden_state = self.online_net(obs_tensor, self.hidden_state, mask_tensor)
            
            action = q_values.argmax(dim=-1).item()
        
        return action, self.hidden_state
    
    def reset_hidden_state(self, batch_size: int = 1):
        self.hidden_state = self.online_net.init_hidden(batch_size)
    
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        legal_mask: np.ndarray,
        info: Dict[str, Any],
    ):
        hidden_for_storage = None
        if self.hidden_state is not None:
            h = self.hidden_state[0].detach()
            c = self.hidden_state[1].detach() if len(self.hidden_state) > 1 else None
            hidden_for_storage = (h, c) if c is not None else (h,)
        
        self.episode_buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            legal_mask=legal_mask,
            info=info,
            hidden_state=hidden_for_storage,
        )
    
    def end_episode(self):
        # convert ep to seqs and add to replay buffer
        sequences = self.episode_buffer.create_sequences(
            sequence_length=SEQUENCE_LENGTH,
            burn_in_length=BURN_IN_LENGTH,
        )
        
        for seq in sequences:
            self.replay_buffer.add(seq)
        
        self.episode_buffer.clear()
        
        self.reset_hidden_state()
    
    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
        
        sequences, indices, weights = self.replay_buffer.sample(BATCH_SIZE)
        batch = collate_sequences(sequences)
        weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        
        if USE_DISTRIBUTIONAL:
            td_errors, loss = self._compute_distributional_loss(batch, weights)
        else:
            td_errors, loss = self._compute_td_loss(batch, weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        
        td_errors_np = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, np.abs(td_errors_np).mean(axis=-1))
        
        self.training_steps += 1
        if self.training_steps % TARGET_UPDATE_FREQ == 0:
            self._update_target_network()
        
        return {
            "loss": loss.item(),
            "td_error": np.abs(td_errors_np).mean(),
            "epsilon": self.current_epsilon,
            "buffer_size": len(self.replay_buffer),
        }
    
    def _compute_td_loss(
        self,
        batch: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        observations = batch["observations"]
        actions = batch["actions"]
        legal_masks = batch["legal_masks"]
        
        n_step_returns = batch["n_step_returns"]
        n_step_next_obs = batch["n_step_next_obs"]
        n_step_dones = batch["n_step_dones"]
        
        batch_size, seq_len, _ = observations.shape
        train_len = TRAIN_LENGTH
        
        if batch["init_hidden_h"] is not None:
            hidden = (batch["init_hidden_h"], batch["init_hidden_c"])
        else:
            hidden = self.online_net.init_hidden(batch_size)
        
        self.online_net.train()
        q_values_seq, _ = self.online_net(observations, hidden, legal_masks)
        
        q_values_train = q_values_seq[:, BURN_IN_LENGTH:, :]
        actions_train = actions[:, BURN_IN_LENGTH:]
        
        q_taken = q_values_train.gather(
            2, actions_train.unsqueeze(-1)
        ).squeeze(-1)
        
        # double DQN: online net selects best action, target net evaluates it
        with torch.no_grad():
            n_step_next_flat = n_step_next_obs.view(batch_size * train_len, -1)
            
            target_hidden = self.target_net.init_hidden(batch_size * train_len)
            
            target_q_flat, _ = self.target_net(n_step_next_flat, target_hidden)
            target_q = target_q_flat.view(batch_size, train_len, -1)
            
            online_hidden = self.online_net.init_hidden(batch_size * train_len)
            online_q_flat, _ = self.online_net(n_step_next_flat, online_hidden)
            online_q = online_q_flat.view(batch_size, train_len, -1)
            
            best_actions = online_q.argmax(dim=-1)
            
            next_q = target_q.gather(
                2, best_actions.unsqueeze(-1)
            ).squeeze(-1)
            
            targets = n_step_returns + self.gamma_n * next_q * (1.0 - n_step_dones)
        
        td_errors = q_taken - targets
        
        loss = (weights.unsqueeze(-1) * td_errors.pow(2)).mean()
        
        return td_errors, loss
    
    def _compute_distributional_loss(
        self,
        batch: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        observations = batch["observations"]
        actions = batch["actions"]
        n_step_returns = batch["n_step_returns"]
        n_step_next_obs = batch["n_step_next_obs"]
        n_step_dones = batch["n_step_dones"]
        
        batch_size, seq_len, _ = observations.shape
        train_len = TRAIN_LENGTH
        
        if batch["init_hidden_h"] is not None:
            hidden = (batch["init_hidden_h"], batch["init_hidden_c"])
        else:
            hidden = self.online_net.init_hidden(batch_size)
        
        last_obs = observations[:, -1, :]
        current_dist, _ = self.online_net.get_q_distribution(last_obs, hidden)
        
        last_action = actions[:, -1]
        current_dist_a = current_dist.gather(
            1, last_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, NUM_ATOMS)
        ).squeeze(1)
        
        with torch.no_grad():
            next_obs = n_step_next_obs[:, -1, :]
            target_hidden = self.target_net.init_hidden(batch_size)
            next_dist, _ = self.target_net.get_q_distribution(next_obs, target_hidden)
            
            online_hidden = self.online_net.init_hidden(batch_size)
            next_q, _ = self.online_net(next_obs.unsqueeze(1), online_hidden)
            next_q = next_q.squeeze(1)
            best_action = next_q.argmax(dim=-1)
            
            next_dist_a = next_dist.gather(
                1, best_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, NUM_ATOMS)
            ).squeeze(1)
            
            reward = n_step_returns[:, -1]
            done = n_step_dones[:, -1]
            
            Tz = reward.unsqueeze(-1) + self.gamma_n * (1 - done.unsqueeze(-1)) * self.support
            Tz = Tz.clamp(V_MIN, V_MAX)
            
            b = (Tz - V_MIN) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            l = l.clamp(0, NUM_ATOMS - 1)
            u = u.clamp(0, NUM_ATOMS - 1)
            
            m = torch.zeros_like(next_dist_a)
            offset = torch.arange(batch_size, device=DEVICE).unsqueeze(-1) * NUM_ATOMS
            
            m.view(-1).index_add_(
                0,
                (offset + l).view(-1),
                (next_dist_a * (u.float() - b)).view(-1)
            )
            m.view(-1).index_add_(
                0,
                (offset + u).view(-1),
                (next_dist_a * (b - l.float())).view(-1)
            )
        
        loss = -(m * current_dist_a.clamp(min=1e-8).log()).sum(dim=-1)
        loss = (weights * loss).mean()
        
        current_q = (current_dist_a * self.support).sum(dim=-1)
        target_q = (m * self.support).sum(dim=-1)
        td_errors = (current_q - target_q).unsqueeze(-1)
        
        return td_errors, loss
    
    def _update_target_network(self):
        if USE_POLYAK:
            for target_param, online_param in zip(
                self.target_net.parameters(),
                self.online_net.parameters()
            ):
                target_param.data.copy_(
                    POLYAK_TAU * online_param.data + (1 - POLYAK_TAU) * target_param.data
                )
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]
    
    def get_policy_action(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
    ) -> int:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask_tensor = torch.tensor(legal_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
            
            hidden = self.online_net.init_hidden(1)
            q_values, _ = self.online_net(obs_tensor, hidden, mask_tensor)
            
            action = q_values.argmax(dim=-1).item()
        
        return action


# adapter for baseline eval harness
class DQNAgentForBaseline:
    
    def __init__(self, agent: DQNAgent, env):
        self.agent = agent
        self.env = env
        self.hidden = None
    
    def select_action(self, game, player: int) -> int:
        self.env.game = game
        self.env.game.current_player = player
        
        obs = self.env._get_observation()
        legal_mask = self.env._get_legal_mask()
        
        main_mask = np.zeros_like(legal_mask)
        valid_moves = game.get_valid_moves(player)
        for m in valid_moves:
            if m < 7:
                main_mask[m] = True
        
        action = self.agent.get_policy_action(obs, main_mask)
        return min(action, 6)
    
    def decide_block(self, game, player: int, block_type: int) -> bool:
        self.env.game = game
        self.env.game.current_player = player
        self.env.phase = "block"
        self.env.pending_action = block_type
        
        obs = self.env._get_observation()
        legal_mask = self.env._get_legal_mask()
        
        action = self.agent.get_policy_action(obs, legal_mask)
        
        return action != 11 and action >= 7 and action <= 10
    
    def decide_challenge(self, game, player: int, claimed_card: int) -> bool:
        self.env.game = game
        self.env.game.current_player = player
        self.env.phase = "challenge_action"
        
        obs = self.env._get_observation()
        legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
        legal_mask[12] = True
        legal_mask[13] = True
        
        action = self.agent.get_policy_action(obs, legal_mask)
        
        return action == 12
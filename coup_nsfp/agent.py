import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE, NUM_ACTIONS, GAMMA,
    BR_EPSILON_START, BR_EPSILON_END, BR_EPSILON_DECAY,
    EPSILON_EXPLORE_START, EPSILON_EXPLORE_END, EPSILON_EXPLORE_DECAY,
    BATCH_SIZE, RL_BUFFER_CAPACITY, SL_BUFFER_CAPACITY,
    LR_BR, LR_AS, TARGET_UPDATE_FREQ,
)
from replay import ReplayBuffer, SLBuffer
from networks import QNetwork, MLP

class NFSPAgent:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim

        self.br_policy = QNetwork(state_dim, NUM_ACTIONS).to(DEVICE)
        self.br_target = QNetwork(state_dim, NUM_ACTIONS).to(DEVICE)
        self.br_target.load_state_dict(self.br_policy.state_dict())
        self.br_optimizer = optim.Adam(self.br_policy.parameters(), lr=LR_BR)

        self.as_policy = MLP(state_dim, NUM_ACTIONS).to(DEVICE)
        self.as_optimizer = optim.Adam(self.as_policy.parameters(), lr=LR_AS)

        self.block_policy = MLP(state_dim + 1, 2).to(DEVICE)
        self.block_optimizer = optim.Adam(self.block_policy.parameters(), lr=LR_AS)

        self.challenge_policy = MLP(state_dim + 1, 2).to(DEVICE)
        self.challenge_optimizer = optim.Adam(self.challenge_policy.parameters(), lr=LR_AS)

        self.rl_buffer = ReplayBuffer(RL_BUFFER_CAPACITY)
        self.sl_buffer = SLBuffer(SL_BUFFER_CAPACITY)
        self.block_buffer = SLBuffer(SL_BUFFER_CAPACITY)
        self.challenge_buffer = SLBuffer(SL_BUFFER_CAPACITY)

        self.br_mode_epsilon = BR_EPSILON_START         
        self.br_explore_epsilon = EPSILON_EXPLORE_START 
        self.total_steps = 0


    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)


    def select_action(self, state: np.ndarray, training: bool = True, valid_actions=None):
        self.total_steps += 1

        # decay epsilon for br mode selection
        self.br_mode_epsilon = max(
            BR_EPSILON_END,
            self.br_mode_epsilon - BR_EPSILON_DECAY
        )

        # choose which policy (br or as) to use
        if training and random.random() < self.br_mode_epsilon:
            mode = "BR"
        else:
            mode = "AS"

        if mode == "BR":
            action = self._select_br_action(state, training, valid_actions)
        else:
            action = self._select_as_action(state, training, valid_actions)

        return action, mode

    def _select_br_action(self, state: np.ndarray, training: bool = True, valid_actions=None) -> int:
        #br network q-learning act
        self.br_explore_epsilon = max(
            EPSILON_EXPLORE_END,
            self.br_explore_epsilon - EPSILON_EXPLORE_DECAY
        )

        if valid_actions is None:
            valid_actions = list(range(NUM_ACTIONS))

        if training and random.random() < self.br_explore_epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            s = self._to_tensor(state)
            q_values = self.br_policy(s)[0]
            
            # Mask invalid actions
            masked_q = q_values.clone()
            invalid_mask = torch.ones(NUM_ACTIONS, dtype=torch.bool, device=DEVICE)
            invalid_mask[valid_actions] = False
            masked_q[invalid_mask] = float('-inf')
            
            return int(masked_q.argmax().item())

    def _select_as_action(self, state: np.ndarray, training: bool = True, valid_actions=None) -> int:
        #avg strat act
        if valid_actions is None:
            valid_actions = list(range(NUM_ACTIONS))
            
        with torch.no_grad():
            s = self._to_tensor(state)
            logits = self.as_policy(s)[0]
            
            # Mask invalid actions
            masked_logits = logits.clone()
            invalid_mask = torch.ones(NUM_ACTIONS, dtype=torch.bool, device=DEVICE)
            invalid_mask[valid_actions] = False
            masked_logits[invalid_mask] = float('-inf')
            
            probs = torch.softmax(masked_logits, dim=0).cpu().numpy()

        if training:
            return int(np.random.choice(NUM_ACTIONS, p=probs))
        else:
            return int(np.argmax(probs))


    def store_transition(self, state, action, reward, next_state, done, mode_used):
        if mode_used == "BR":
            self.rl_buffer.push(state, action, reward, next_state, done)
            self.sl_buffer.push(state, action)
        

    def train_step(self):
        #br and as
        if len(self.rl_buffer) >= BATCH_SIZE:
            self._train_br()

        if len(self.sl_buffer) >= BATCH_SIZE:
            self._train_as()

    def _train_br(self):
        #br newwork q-learning
        batch = self.rl_buffer.sample(BATCH_SIZE)

        state_batch = torch.tensor(
            np.stack(batch.state), dtype=torch.float32, device=DEVICE
        )
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state_batch = torch.tensor(
            np.stack(batch.next_state), dtype=torch.float32, device=DEVICE
        )
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Current Q-values
        q_values = self.br_policy(state_batch).gather(1, action_batch)

        # Target Q-values using target network
        with torch.no_grad():
            max_next_q = self.br_target(next_state_batch).max(dim=1, keepdim=True)[0]
            target_q = reward_batch + (1.0 - done_batch) * GAMMA * max_next_q

        # MSE
        loss = nn.MSELoss()(q_values, target_q)

        self.br_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.br_policy.parameters(), 5.0)
        self.br_optimizer.step()

        # update target network
        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.br_target.load_state_dict(self.br_policy.state_dict())

    def _train_as(self):
        #avg strat
        states, actions = self.sl_buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)

        logits = self.as_policy(states)
        loss = nn.CrossEntropyLoss()(logits, actions)

        self.as_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.as_policy.parameters(), 5.0)
        self.as_optimizer.step()

    def select_block_action(self, state: np.ndarray, block_type: int, training: bool = True):
        state_with_type = np.concatenate([state, np.array([float(block_type)], dtype=np.float32)])
        
        with torch.no_grad():
            s = torch.tensor(state_with_type, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = self.block_policy(s)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        if training:
            action = int(np.random.choice(2, p=probs))
        else:
            action = int(np.argmax(probs))
        
        return action, state_with_type

    def select_challenge_action(self, state: np.ndarray, claimed_card: int, training: bool = True):
        if isinstance(claimed_card, (list, tuple)):
            claimed_card = claimed_card[0]
        
        state_with_card = np.concatenate([state, np.array([float(claimed_card)], dtype=np.float32)])
        
        with torch.no_grad():
            s = torch.tensor(state_with_card, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = self.challenge_policy(s)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        if training:
            action = int(np.random.choice(2, p=probs))
        else:
            action = int(np.argmax(probs))
        
        return action, state_with_card

    def store_block_transition(self, state_with_type, action):
        self.block_buffer.push(state_with_type, action)

    def store_challenge_transition(self, state_with_card, action):
        self.challenge_buffer.push(state_with_card, action)

    def train_block(self):
        if len(self.block_buffer) >= BATCH_SIZE:
            states, actions = self.block_buffer.sample(BATCH_SIZE)
            states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)

            logits = self.block_policy(states)
            loss = nn.CrossEntropyLoss()(logits, actions)

            self.block_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.block_policy.parameters(), 5.0)
            self.block_optimizer.step()

    def train_challenge(self):
        if len(self.challenge_buffer) >= BATCH_SIZE:
            states, actions = self.challenge_buffer.sample(BATCH_SIZE)
            states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
            actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)

            logits = self.challenge_policy(states)
            loss = nn.CrossEntropyLoss()(logits, actions)

            self.challenge_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.challenge_policy.parameters(), 5.0)
            self.challenge_optimizer.step()

    def save(self, path_prefix: str):
        torch.save(self.br_policy.state_dict(), f"{path_prefix}_q.pt")
        torch.save(self.br_target.state_dict(), f"{path_prefix}_q_target.pt")
        torch.save(self.as_policy.state_dict(), f"{path_prefix}_avg_strategy.pt")
        torch.save(self.block_policy.state_dict(), f"{path_prefix}_block.pt")
        torch.save(self.challenge_policy.state_dict(), f"{path_prefix}_challenge.pt")

    def load(self, path_prefix: str):
        self.br_policy.load_state_dict(
            torch.load(f"{path_prefix}_q.pt", map_location=DEVICE)
        )
        self.br_target.load_state_dict(
            torch.load(f"{path_prefix}_q_target.pt", map_location=DEVICE)
        )
        self.as_policy.load_state_dict(
            torch.load(f"{path_prefix}_avg_strategy.pt", map_location=DEVICE)
        )
        self.block_policy.load_state_dict(
            torch.load(f"{path_prefix}_block.pt", map_location=DEVICE)
        )
        self.challenge_policy.load_state_dict(
            torch.load(f"{path_prefix}_challenge.pt", map_location=DEVICE)
        )
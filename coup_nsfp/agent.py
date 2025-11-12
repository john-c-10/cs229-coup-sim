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

        #target is a delayed copy of br_policy that is used to compute desired target value (r + max a' from reached state s')
        self.br_target = QNetwork(state_dim, NUM_ACTIONS).to(DEVICE)

        self.br_target.load_state_dict(self.br_policy.state_dict())
        self.br_optimizer = optim.Adam(self.br_policy.parameters(), lr=LR_BR)

        self.as_policy = MLP(state_dim, NUM_ACTIONS).to(DEVICE)
        self.as_optimizer = optim.Adam(self.as_policy.parameters(), lr=LR_AS)

        self.rl_buffer = ReplayBuffer(RL_BUFFER_CAPACITY)
        self.sl_buffer = SLBuffer(SL_BUFFER_CAPACITY)

        self.br_mode_epsilon = BR_EPSILON_START         
        self.br_explore_epsilon = EPSILON_EXPLORE_START 
        self.total_steps = 0


    #take numpy array and turn into tensor of (1, state_dim)
    #needed for pytorch and backprop

    def _to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)


    def select_action(self, state: np.ndarray, training: bool = True):
        self.total_steps += 1

        #goes from .1 to .02 over time
        self.br_mode_epsilon = max(
            BR_EPSILON_END,
            self.br_mode_epsilon - BR_EPSILON_DECAY
        )

        # choose which head (BR or AS) to use
        if training and random.random() < self.br_mode_epsilon:
            mode = "BR"
        else:
            mode = "AS"

        if mode == "BR":
            action = self._select_br_action(state, training)
        else:
            action = self._select_as_action(state, training)

        return action, mode

    def _select_br_action(self, state: np.ndarray, training: bool = True) -> int:
        #this is random vs greedy factor
        self.br_explore_epsilon = max(
            EPSILON_EXPLORE_END,
            self.br_explore_epsilon - EPSILON_EXPLORE_DECAY
        )

        if training and random.random() < self.br_explore_epsilon:
            return random.randrange(NUM_ACTIONS)

        with torch.no_grad():
            s = self._to_tensor(state)
            q_values = self.br_policy(s)
            #q_values shape is (1, num_actions)
            return int(q_values.argmax(dim=1).item())

    def _select_as_action(self, state: np.ndarray, training: bool = True) -> int:
        with torch.no_grad():
            s = self._to_tensor(state)
            logits = self.as_policy(s)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        if training:
            return int(np.random.choice(NUM_ACTIONS, p=probs))
        else:
            #deterministic choice in order for predictability like the best according to history
            return int(np.argmax(probs))


    def store_transition(self, state, action, reward, next_state, done, mode_used):
        if mode_used == "BR":
            self.rl_buffer.push(state, action, reward, next_state, done)
            self.sl_buffer.push(state, action)
        

    def train_step(self):

        #once reaches certain size, we want to train on them and update our model

        if len(self.rl_buffer) >= BATCH_SIZE:
            self._train_br()

        if len(self.sl_buffer) >= BATCH_SIZE:
            self._train_as()

    def _train_br(self):

        #sample from buffer
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

        q_values = self.br_policy(state_batch).gather(1, action_batch)

        with torch.no_grad():
            max_next_q = self.br_target(next_state_batch).max(dim=1, keepdim=True)[0]
            target_q = reward_batch + (1.0 - done_batch) * GAMMA * max_next_q
            #target_q = r + γ max_a' Q_target(s', a')


        #real-valued scalars so mse makes sense
        loss = nn.MSELoss()(q_values, target_q)

        #clears existing grad
        self.br_optimizer.zero_grad()
        #new grad that is based on q_values forward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.br_policy.parameters(), 5.0)
        self.br_optimizer.step()

        # target update
        #we want to update target every once in a while
        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.br_target.load_state_dict(self.br_policy.state_dict())

    def _train_as(self):
        states, actions = self.sl_buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)

        logits = self.as_policy(states)

        #standard loss for classification
        loss = nn.CrossEntropyLoss()(logits, actions)

        #trying to minimize difference between predicted action and actual action

        self.as_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.as_policy.parameters(), 5.0)
        self.as_optimizer.step()


    def save(self, path_prefix: str):
        torch.save(self.br_policy.state_dict(), f"{path_prefix}_br.pt")
        torch.save(self.br_target.state_dict(), f"{path_prefix}_br_target.pt")
        torch.save(self.as_policy.state_dict(), f"{path_prefix}_as.pt")

    def load(self, path_prefix: str):
        self.br_policy.load_state_dict(
            torch.load(f"{path_prefix}_br.pt", map_location=DEVICE)
        )
        self.br_target.load_state_dict(
            torch.load(f"{path_prefix}_br_target.pt", map_location=DEVICE)
        )
        self.as_policy.load_state_dict(
            torch.load(f"{path_prefix}_as.pt", map_location=DEVICE)
        )

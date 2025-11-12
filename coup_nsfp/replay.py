#buffer definitions that are just tracking recent x entries for training

import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))


#used for the Q learning and only keeps track of recent maxlen entries


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)




class SLBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action):
        self.buffer.append((state, action))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions = zip(*batch)
        return np.stack(states), np.array(actions, dtype=np.int64)

    def __len__(self):
        return len(self.buffer)

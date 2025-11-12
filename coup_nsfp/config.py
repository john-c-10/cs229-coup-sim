#params used throughout the code 
#currently the br and epsilon greedy-exploration params are a bit arbitrary but seem to be somewhat standard


import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Card indices: [Duke, Assassin, Captain, Ambassador, Contessa]
NUM_CARD_TYPES = 5

ACTIONS = [
    "INCOME",       # 0
    "FOREIGN_AID", # 1
    "TAX",         # 2 (claim Duke)
    "STEAL",       # 3 (claim Captain)
    "EXCHANGE",    # 4 (claim Ambassador)
    "ASSASSINATE", # 5 (claim Assassin)
    "COUP",        # 6
]
NUM_ACTIONS = len(ACTIONS)

# Discount factor
GAMMA = 0.95

# NFSP “mode” mixing: probability of using BR instead of AS
BR_EPSILON_START = 0.1
BR_EPSILON_END = 0.02
BR_EPSILON_DECAY = 1e-5

# Epsilon-greedy inside BR DQN
EPSILON_EXPLORE_START = 0.2
EPSILON_EXPLORE_END = 0.05
EPSILON_EXPLORE_DECAY = 1e-5

# Buffers / training
BATCH_SIZE = 64
RL_BUFFER_CAPACITY = 100_000
SL_BUFFER_CAPACITY = 100_000
LR_BR = 1e-4
LR_AS = 1e-4
TARGET_UPDATE_FREQ = 1000  # steps

MAX_TURNS_PER_GAME = 50

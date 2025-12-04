import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAME-SPECIFIC CONSTANTS

# card types: Duke (0), Assassin (1), Ambassador (2), Captain (3), Contessa (4)
NUM_CARD_TYPES = 5
CARD_NAMES = ["Duke", "Assassin", "Ambassador", "Captain", "Contessa"]

ACTIONS = [
    "INCOME", # 0 (no claim, cannot be blocked)
    "FOREIGN_AID", # 1 (no claim, can be blocked by Duke)
    "TAX", # 2 (claim Duke, gain 3 coins)
    "STEAL", # 3 (claim Captain, steal up to 2 coins)
    "EXCHANGE", # 4 (claim Ambassador, exchange cards)
    "ASSASSINATE", # 5 (claim Assassin, costs 3 coins)
    "COUP", # 6 (no claim, costs 7 coins, cannot be blocked)
]
NUM_MAIN_ACTIONS = len(ACTIONS)

BLOCK_ACTIONS = [
    "BLOCK_FOREIGN_AID", # 7 (claim Duke to block foreign aid)
    "BLOCK_ASSASSINATION", # 8 (claim Contessa to block assassination)
    "BLOCK_STEAL_CAPTAIN", # 9 (claim Captain to block steal)
    "BLOCK_STEAL_AMBASSADOR", # 10 (claim Ambassador to block steal)
    "DECLINE_BLOCK", # 11 (choose not to block)
]
NUM_BLOCK_ACTIONS = len(BLOCK_ACTIONS)

CHALLENGE_ACTIONS = [
    "CHALLENGE", # 12 (challenge opponent's claim)
    "ACCEPT", # 13 (accept opponent's claim (aka don't challenge))
]
NUM_CHALLENGE_ACTIONS = len(CHALLENGE_ACTIONS)

ALL_ACTIONS = ACTIONS + BLOCK_ACTIONS + CHALLENGE_ACTIONS
NUM_ACTIONS = len(ALL_ACTIONS)

# OBSERVATION SPACE CONFIG

HISTORY_LENGTH = 10
ACTION_ENCODING_DIM = NUM_ACTIONS + 2 # action (one-hot) + actor + target

PUBLIC_FEATURES = 4 + NUM_CARD_TYPES # coins, influence, discard
PRIVATE_FEATURES = NUM_CARD_TYPES # own cards
HISTORY_FEATURES = HISTORY_LENGTH * ACTION_ENCODING_DIM
TURN_FEATURES = 2

OBS_DIM = PUBLIC_FEATURES + PRIVATE_FEATURES + HISTORY_FEATURES + TURN_FEATURES

# NETWORK ARCHITECTURE

CARD_EMBED_DIM = 16
ACTION_EMBED_DIM = 16

TORSO_HIDDEN_DIMS = (128, 128)

USE_LSTM = True # false for GRU
RNN_HIDDEN_DIM = 128
RNN_NUM_LAYERS = 1

VALUE_HIDDEN_DIM = 64
ADVANTAGE_HIDDEN_DIM = 64

USE_NOISY_NETS = False
NOISY_STD_INIT = 0.5

USE_DISTRIBUTIONAL = False
NUM_ATOMS = 51
V_MIN = -10.0
V_MAX = 10.0

# REPLAY BUFFER CONFIG (R2D2 STYLE)

SEQUENCE_LENGTH = 20 # total sequence length
BURN_IN_LENGTH = 5 # burn-in steps to warm up LSTM hidden state
TRAIN_LENGTH = SEQUENCE_LENGTH - BURN_IN_LENGTH # steps to train on

USE_PRIORITIZED_REPLAY = True
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_FRAMES = 100_000
PER_EPSILON = 1e-6

REPLAY_BUFFER_CAPACITY = 50_000

# TRAINING HYPERPARAMS

GAMMA = 0.99

N_STEP = 3

LEARNING_RATE = 1e-4
ADAM_EPS = 1e-8

BATCH_SIZE = 32

TARGET_UPDATE_FREQ = 1000
USE_POLYAK = False
POLYAK_TAU = 0.005

GRAD_CLIP_NORM = 10.0

CLIP_REWARDS = True
REWARD_CLIP_MIN = -1.0
REWARD_CLIP_MAX = 1.0

# EXPLORATION

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_FRAMES = 100_000

# TRAINING LOOP CONFIG

NUM_EPISODES = 50_000
MAX_STEPS_PER_EPISODE = 100

WARMUP_STEPS = 1000

TRAIN_FREQ = 4

EVAL_FREQ = 1000
EVAL_GAMES = 100

LOG_FREQ = 100

SAVE_FREQ = 5000
CHECKPOINT_DIR = "checkpoints/dqn"

# TIME PENALTY CONFIG

USE_TIME_PENALTY = False
TIME_PENALTY = -0.001
STALL_THRESHOLD = 50

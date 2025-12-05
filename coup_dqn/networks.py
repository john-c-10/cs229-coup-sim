import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import (
    DEVICE,
    NUM_CARD_TYPES,
    NUM_ACTIONS,
    CARD_EMBED_DIM,
    ACTION_EMBED_DIM,
    TORSO_HIDDEN_DIMS,
    USE_LSTM,
    RNN_HIDDEN_DIM,
    RNN_NUM_LAYERS,
    VALUE_HIDDEN_DIM,
    ADVANTAGE_HIDDEN_DIM,
    USE_NOISY_NETS,
    NOISY_STD_INIT,
    USE_DISTRIBUTIONAL,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    OBS_DIM,
)

# parametric noise layer (replaces eps-greedy exploration)
class NoisyLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, std_init: float = NOISY_STD_INIT):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class Torso(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = TORSO_HIDDEN_DIMS):
        super().__init__()
        
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        self.output_dim = last_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# handles partial observability via LSTM/GRU
class RecurrentCore(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = RNN_HIDDEN_DIM,
        num_layers: int = RNN_NUM_LAYERS,
        use_lstm: bool = USE_LSTM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        
        output, hidden = self.rnn(x, hidden)
        return output, hidden
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        if self.use_lstm:
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h, c)
        return (h,)

# dueling head (here there are separate value and advantage streams)
class DuelingHead(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int = NUM_ACTIONS,
        value_hidden_dim: int = VALUE_HIDDEN_DIM,
        advantage_hidden_dim: int = ADVANTAGE_HIDDEN_DIM,
        use_noisy: bool = USE_NOISY_NETS,
    ):
        super().__init__()
        self.num_actions = num_actions
        
        Linear = NoisyLinear if use_noisy else nn.Linear
        
        # value stream
        self.value_fc1 = Linear(input_dim, value_hidden_dim)
        self.value_fc2 = Linear(value_hidden_dim, 1)
        
        # advantage stream
        self.advantage_fc1 = Linear(input_dim, advantage_hidden_dim)
        self.advantage_fc2 = Linear(advantage_hidden_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values
    
    def reset_noise(self):
        if USE_NOISY_NETS:
            self.value_fc1.reset_noise()
            self.value_fc2.reset_noise()
            self.advantage_fc1.reset_noise()
            self.advantage_fc2.reset_noise()

class DistributionalDuelingHead(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int = NUM_ACTIONS,
        value_hidden_dim: int = VALUE_HIDDEN_DIM,
        advantage_hidden_dim: int = ADVANTAGE_HIDDEN_DIM,
        num_atoms: int = NUM_ATOMS,
        v_min: float = V_MIN,
        v_max: float = V_MAX,
        use_noisy: bool = USE_NOISY_NETS,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        Linear = NoisyLinear if use_noisy else nn.Linear
        
        # value stream
        self.value_fc1 = Linear(input_dim, value_hidden_dim)
        self.value_fc2 = Linear(value_hidden_dim, num_atoms)
        
        # advantage stream
        self.advantage_fc1 = Linear(input_dim, advantage_hidden_dim)
        self.advantage_fc2 = Linear(advantage_hidden_dim, num_actions * num_atoms)
        
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, num_atoms)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value).view(batch_size, 1, self.num_atoms)
        
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage).view(
            batch_size, self.num_actions, self.num_atoms
        )
        
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        q_dist = self.forward(x)
        q_values = (q_dist * self.support).sum(dim=-1)
        return q_values
    
    def reset_noise(self):
        if USE_NOISY_NETS:
            self.value_fc1.reset_noise()
            self.value_fc2.reset_noise()
            self.advantage_fc1.reset_noise()
            self.advantage_fc2.reset_noise()

# the full network is torso MLP -> recurrent core -> dueling head
class DuelingDRQN(nn.Module):
    
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        torso_hidden_dims: Tuple[int, ...] = TORSO_HIDDEN_DIMS,
        rnn_hidden_dim: int = RNN_HIDDEN_DIM,
        rnn_num_layers: int = RNN_NUM_LAYERS,
        use_lstm: bool = USE_LSTM,
        use_distributional: bool = USE_DISTRIBUTIONAL,
        use_noisy: bool = USE_NOISY_NETS,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.use_distributional = use_distributional
        
        self.torso = Torso(obs_dim, torso_hidden_dims)
        
        self.rnn = RecurrentCore(
            input_dim=self.torso.output_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            use_lstm=use_lstm,
        )
        
        if use_distributional:
            self.head = DistributionalDuelingHead(
                input_dim=rnn_hidden_dim,
                num_actions=num_actions,
                use_noisy=use_noisy,
            )
        else:
            self.head = DuelingHead(
                input_dim=rnn_hidden_dim,
                num_actions=num_actions,
                use_noisy=use_noisy,
            )
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        legal_actions_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)
            if legal_actions_mask is not None:
                legal_actions_mask = legal_actions_mask.unsqueeze(1)
        
        batch_size, seq_len, _ = obs.shape
        
        obs_flat = obs.view(batch_size * seq_len, -1)
        features_flat = self.torso(obs_flat)
        features = features_flat.view(batch_size, seq_len, -1)
        
        rnn_out, hidden = self.rnn(features, hidden)
        
        if self.use_distributional:
            rnn_flat = rnn_out.view(batch_size * seq_len, -1)
            q_values_flat = self.head.get_q_values(rnn_flat)
            q_values = q_values_flat.view(batch_size, seq_len, -1)
        else:
            q_values = self.head(rnn_out)
        
        # mask actions that are illegal with -inf so never selected
        if legal_actions_mask is not None:
            illegal_mask = ~legal_actions_mask
            q_values = q_values.masked_fill(illegal_mask, float("-inf"))
        
        if single_step:
            q_values = q_values.squeeze(1)
        
        return q_values, hidden
    
    def get_q_distribution(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if not self.use_distributional:
            raise ValueError("get_q_distribution only available in distributional mode")
        
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)
        
        batch_size, seq_len, _ = obs.shape
        
        obs_flat = obs.view(batch_size * seq_len, -1)
        features_flat = self.torso(obs_flat)
        features = features_flat.view(batch_size, seq_len, -1)
        
        rnn_out, hidden = self.rnn(features, hidden)
        
        rnn_last = rnn_out[:, -1, :]
        q_dist = self.head(rnn_last)
        
        return q_dist, hidden
    
    def reset_noise(self):
        self.head.reset_noise()
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        return self.rnn.init_hidden(batch_size, DEVICE)

def create_network(
    obs_dim: int = OBS_DIM,
    num_actions: int = NUM_ACTIONS,
) -> DuelingDRQN:
    return DuelingDRQN(
        obs_dim=obs_dim,
        num_actions=num_actions,
        use_distributional=USE_DISTRIBUTIONAL,
        use_noisy=USE_NOISY_NETS,
    ).to(DEVICE)
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from torch.distributions import Beta, Normal

class Factor_Reward_Model(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_layers=5, hidden_dim=64, device='cuda'):
        super().__init__()
        self.n_layers = n_layers
        
        # 10個の因子に対応した入力次元
        # input_dim should be 10 for the new evaluation function
        
        # デバイス設定をより柔軟に（DRP用調整）
        if isinstance(device, str):
            if device == 'cuda' and not torch.cuda.is_available():
                device = 'cpu'
            self.device = torch.device(device)
        else:
            self.device = device
        
        if n_layers == 1:
            self.model = nn.Linear(input_dim, output_dim)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers-2)],
                nn.Linear(hidden_dim, output_dim)
            )
        
        self.to(self.device)
        # self.apply(self.init_weights)

    def forward(self, x):
        # x should be shape (batch_size, 11) for 11 factors
        return self.model(x)
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
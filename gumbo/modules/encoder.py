import torch as th
import torch.nn as nn


class IdentityEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def build(self, obs_spec):
        return obs_spec.shape.numel()
    
    def forward(self, obs):
        return obs


class FlattenEncoder(nn.Module):
    
    model: nn.Module

    def __init__(self):
        super().__init__()
    
    def build(self, obs_spec):
        self.model = nn.Flatten(
            start_dim=-len(obs_spec.shape))
        return obs_spec.shape.numel()
    
    def forward(self, obs):
        return self.model(obs)
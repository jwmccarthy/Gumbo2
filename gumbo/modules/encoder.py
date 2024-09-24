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
        self.model = nn.Flatten(start_dim=-len(obs_spec.shape))
        return obs_spec.shape.numel()
    
    def forward(self, obs):
        if obs.dtype == th.uint8:
            obs = obs.float() / 255.0
        return self.model(obs)
    

class ImageEncoder(nn.Module):

    model: nn.Module

    def __init__(self):
        pass

    def build(self, obs_spec):
        pass

    def forward(self, obs):
        obs = obs.float() / 255.0
        return self.model(obs)
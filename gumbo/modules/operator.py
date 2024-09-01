import torch as th
import torch.nn as nn


class ValueOperator(nn.Module):
    """
    Value operator for evaluating observations
    """

    model: nn.Module

    def __init__(self, encoder, body):
        super().__init__()
        self.encoder = encoder
        self.body = body

    def build(self, obs_spec):
        feat_dims = self.encoder.build(obs_spec)
        self.body.build(feat_dims, 1)
        self.model = nn.Sequential(self.encoder, self.body)
        self.shape = obs_spec.shape
        return self
    
    def forward(self, obs):
        val = self.model(obs)
        if val.ndim > 1:
            val = th.squeeze(val, -1)
        return val
import torch as th
import torch.nn as nn


def dims_to_sequential(in_dim, hidden_dims, out_dim, act_func):
    modules = []
    last_dim = in_dim
    for next_dim in hidden_dims:
        modules.extend([
            nn.Linear(last_dim, next_dim), act_func()])
        last_dim = next_dim
    modules.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*modules)


class MLP(nn.Module):

    model: nn.Module

    def __init__(self, hidden_dim=(64, 64), act_fn=nn.Tanh):
        super().__init__()
        self.act_fn = act_fn
        self.hidden_dim = hidden_dim

    def build(self, in_dim, out_dim):
        self.model = dims_to_sequential(
            in_dim, self.hidden_dim, out_dim, self.act_fn)
        return out_dim
    
    def forward(self, x):
        return self.model(x)
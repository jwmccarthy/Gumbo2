import torch as th
import torch.nn as nn

from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

from abc import abstractmethod


class BasePolicy(nn.Module):
    """
    Base class for policy implementations
    """

    model: nn.Module

    def __init__(self, encoder, body):
        super().__init__()
        self.encoder = encoder
        self.body = body

    def build(self, obs_spec, act_spec):
        feat_dims = self.encoder.build(obs_spec)
        self.body.build(feat_dims, act_spec.logit)
        self.model = nn.Sequential(self.encoder, self.body)
        return self

    @abstractmethod
    def dist(self, obs):
        ...

    @abstractmethod
    def action(self, obs):
        ...

    @abstractmethod
    def sample(self, obs):
        ...

    def log_probs(self, obs, act):
        return self.dist(obs).log_prob(act)
    
    def entropy(self, obs):
        return self.dist(obs).entropy()

    def forward(self, obs, sample=True):
        if sample:
            return self.sample(obs)
        return self.action(obs)
    

class CategoricalPolicy(BasePolicy):
    """
    Policy for simple discrete action spaces
    """

    def __init__(self, encoder, body):
        super().__init__(encoder, body)

    def dist(self, obs):
        return Categorical(logits=self.model(obs))
    
    def action(self, obs):
        return th.argmax(self.model(obs), dim=-1)
    
    def sample(self, obs):
        return self.dist(obs).sample()
    

class DiagonalGaussianPolicy(BasePolicy):
    """
    Basic implementation of a diagonal Gaussian policy for continous control
    """

    def __init__(self, encoder, body):
        super().__init__(encoder, body)

    def build(self, obs_spec, act_spec):
        super().build(obs_spec, act_spec)
        self.covmat = th.eye(act_spec.logit)
        return self

    def dist(self, obs):
        return MultivariateNormal(
            loc=self.model(obs), covariance_matrix=self.covmat)
    
    def action(self, obs):
        return self.model(obs)
    
    def sample(self, obs):
        return self.dist(obs).sample()
from gumbo.env.spaces import BoxSpace
from gumbo.env.spaces import DiscreteSpace

from gumbo.modules.policy import CategoricalPolicy
from gumbo.modules.policy import DiagonalGaussianPolicy


def policy_from_env(env):
    if isinstance(env.act_spec, BoxSpace):
        return DiagonalGaussianPolicy
    if isinstance(env.act_spec, DiscreteSpace):
        return CategoricalPolicy
    raise NotImplementedError(
        f"Policy for {env.act_spec} not implemented")
import torch as th

from gymnasium import spaces
from gymnasium.spaces import Space

from typing import List
from typing import SupportsFloat

from gumbo.types import TorchSpec
from gumbo.types import torch_dtypes


def _torch_spec_from_space(space: Space, device="cpu"):
    if isinstance(space, spaces.Box):
        return BoxSpace(space, device=device)
    if isinstance(space, spaces.Discrete):
        return DiscreteSpace(space, device=device)
    if isinstance(space, spaces.MultiDiscrete):
        return MultiDiscreteSpace(space, device=device)
    if isinstance(space, spaces.MultiBinary):
        return MultiBinarySpace(space, device=device)
    if isinstance(space, spaces.Tuple):
        return TorchTupleSpace(space, device=device)
    if isinstance(space, spaces.Dict):
        return TorchDictSpace(space, device=device)
    raise TypeError(f"Unsupported space type {type(space)}")


def _logit_dim(space: spaces.Space):
    if isinstance(space, spaces.Box):
        return space.shape[0]
    if isinstance(space, spaces.Discrete):
        return space.n
    if isinstance(space, spaces.MultiDiscrete):
        return sum(space.nvec)
    if isinstance(space, spaces.MultiBinary):
        return space.n
    if isinstance(space, spaces.Tuple):
        return sum(_logit_dim(s) for s in space.spaces)
    if isinstance(space, spaces.Dict):
        return sum(_logit_dim(s) for s in space.spaces.values())
    raise TypeError(f"Unsupported space type {type(space)}")


class TorchSpace(TorchSpec):

    space: Space

    def __init__(self, space, device="cpu"):
        self.space = space
        super().__init__(
            shape=th.Size(space.shape),
            dtype=torch_dtypes[space.dtype.type],
            device=device
        )

    def sample(self):
        return th.as_tensor(self.space.sample(), dtype=self.dtype)
    
    def contains(self, x):
        return self.space.contains(x.cpu().numpy())
    
    @property
    def logit(self):
        return _logit_dim(self.space)
    

class BoxSpace(TorchSpace):

    low:  th.Tensor | SupportsFloat
    high: th.Tensor | SupportsFloat

    def __init__(self, space, device="cpu"):
        super().__init__(space, device=device)
        self.low = th.as_tensor(space.low)
        self.high = th.as_tensor(space.high)

    
class DiscreteSpace(TorchSpace):

    n: int

    def __init__(self, space, device="cpu"):
        super().__init__(space, device=device)
        self.n = space.n


class MultiDiscreteSpace(TorchSpace):

    nvec: List[int]

    def __init__(self, space, device="cpu"):
        super().__init__(space, device=device)
        self.nvec = space.nvec


class MultiBinarySpace(TorchSpace):

    n: int

    def __init__(self, space, device="cpu"):
        super().__init__(space, device=device)
        self.n = space.n


class TorchTupleSpace:

    spaces: List[TorchSpace]

    def __init__(self, space, device="cpu"):
        self.spaces = [
            TorchSpace(s, device=device) for s in space.spaces]

    def sample(self):
        return (s.sample() for s in self.spaces)
    
    def contains(self, x):
        return all(s.contains(v) for s, v in zip(self.spaces, x))
    

class TorchDictSpace:

    spaces: dict[str, TorchSpace]

    def __init__(self, space, device):
        self.spaces = {
            k: TorchSpace(v, device=device) for k, v in space.spaces.items()}

    def sample(self):
        return {k: s.sample() for k, s in self.spaces.items()}
    
    def contains(self, x):
        return all(s.contains(x[k]) for k, s in self.spaces.items())
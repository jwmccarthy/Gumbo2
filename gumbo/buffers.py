import torch as th

from typing import List

from gumbo.types import Index, Device
from gumbo.bundle import TensorBundle
from gumbo.bundle import BundleSubset


# TODO: on-policy & off-policy implementations?
# TODO: handle on/off-policy via the collector?
# TODO: "add" method to account for index type?
class Buffer(TensorBundle):
    """
    TensorBundle with additional methods for managing experience data
    """

    def __init__(self, data: dict, device: Device = "cpu"):
        super().__init__(data, device=device)

    # TODO: create compositive spec with its own device attribute
    @classmethod
    def from_spec(cls, spec: dict, size: int):
        data = {}
        for key, val in spec.items():
            shape = (size, *val.shape)
            data[key] = th.empty(
                shape, dtype=val.dtype, device=val.device)
        return cls(data, device=val.device)
    
    def copy(self):
        return Buffer(self._data, device=self.device)
    

class EpisodicBuffer(Buffer):
    """
    Buffer that stores episodes as views of its contiguous data
    """

    _index: int
    _episodes: List[dict]

    def __init__(self, data: dict, device: Device = "cpu"):
        super().__init__(data, device=device)
        self._index = 0
        self._episodes = []

    def add_episode(self, info: dict):
        episode = BundleSubset(
            self, info.pop("idx"))
        for key, val in info.items():
            setattr(episode, key, val)
        self._episodes.append(episode)

    @property
    def episodes(self):
        return self._episodes[:]
    
    def copy(self):
        buffer = EpisodicBuffer(
            self._data, device=self.device)
        buffer._episodes = self.episodes
        return buffer
    
    def clear(self):
        self._index = 0
        self._episodes.clear()
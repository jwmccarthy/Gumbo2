import torch as th

from typing import Self

from collections import defaultdict

from gumbo.data.types import Index
from gumbo.data.types import Device


class TensorBundle:
    """
    Nested collection of PyTorch tensors with common Tensor operations
    """

    def __init__(self, data: dict, device: Device = "cpu"):
        self._data = {}
        self.device = device
        for key, val in data.items():
            self._data[key] = self._parse_data(val)

    def _parse_data(self, data):
        if isinstance(data, dict):
            return TensorBundle(data, device=self.device)
        return data.to(self.device)
    
    def __getitem__(self, idx: Index):
        return TensorBundle(
            {key: val[idx] for key, val in self._data.items()},
            device=self.device
        )
    
    def __setitem__(self, idx: Index, val: dict | Self):
        if isinstance(val, TensorBundle):
            val = val._data
        for key, data in val.items():
            self._data[key][idx] = self._parse_data(data)

    def __getattr__(self, key: str):
        if key in self._data:
            return self._data[key]
        return self.__getattribute__(key)
    
    def __len__(self):
        return len(next(iter(self._data.values())))
    
    def __contains__(self, key: str):
        if key in self._data:
            return True
        for val in self._data.values():
            if isinstance(val, TensorBundle) and key in val:
                return True
        return False
    
    def set(self, **kwargs):
        for key, val in kwargs.items():
            self._data[key] = self._parse_data(val)
    
    def to(self, device: Device):
        self.device = device
        for key, val in self._data.items():
            self._data[key] = val.to(device)
        return self
    

class TensorSubset(TensorBundle):
    """
    Reference to a subset of a TensorBundle
    """

    _data: TensorBundle
    _index: Index

    def __init__(self, data: TensorBundle, index: Index):
        self._data = data
        self._index = index

    def __getitem__(self, idx: Index):
        return self._data[self._index][idx]
    
    def __setitem__(self, idx: Index, val: dict | Self):
        self._data[self._index][idx] = val
    
    def __getattr__(self, key: str):
        if key in self._data:
            return getattr(self._data[self._index], key)
        return self.__getattribute__(key)
    
    def __len__(self):
        return len(self._data[self._index])
    
    def __contains__(self, key: str):
        return key in self._data
    
    @property
    def device(self):
        return self._data.device

    def point(self, data: TensorBundle):
        self._data = data
        return self

    def set(self, **kwargs):
        self._data.set(**kwargs)


class ListBundle:
    """
    TensorBundle-like functionality for lists w/ defaultdict behavior
    """

    def __init__(self, data: dict = None):
        self._data = {}
        if data is None: data = {}
        for key, val in data.items():
            self._data[key] = self._parse_data(val)

    def _parse_data(self, data):
        if isinstance(data, dict):
            return ListBundle(data)
        return data
    
    def __setitem__(self, idx: Index, val: dict | Self):
        if isinstance(val, ListBundle):
            val = val._data
        for key, data in val.items():
            self._data[key][idx] = self._parse_data(data)

    def __getitem__(self, idx: Index):
        return ListBundle(
            {key: val[idx] for key, val in self._data.items()})
    
    def __getattr__(self, key: str):
        if key in self._data:
            return self._data[key]
        return self.__getattribute__(key)
    
    def __len__(self):
        return len(next(iter(self._data.values())))
    
    def __contains__(self, key: str):
        if key in self._data:
            return True
        for val in self._data.values():
            if isinstance(val, ListBundle) and key in val:
                return True
        return False
    
    def set(self, **kwargs):
        for key, val in kwargs.items():
            self._data[key] = self._parse_data(val)

    def append(self, data: dict):
        if isinstance(data, ListBundle):
            data = data._data
        for key, val in data.items():
            if key in self._data:
                self._data[key].append(val)
            else:
                if isinstance(val, dict):
                    self._data[key] = ListBundle().append(val)
                else:
                    self._data[key] = [val]
        return self
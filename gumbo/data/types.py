import numpy as np
import torch as th

from dataclasses import dataclass

from typing import Iterable


torch_dtypes = {
    np.uint8      : th.uint8,
    np.int8       : th.int8,
    np.int16      : th.int16,
    np.int32      : th.int32,
    np.int64      : th.int64,
    np.float16    : th.float16,
    np.float32    : th.float32,
    np.float64    : th.float64,
    np.complex64  : th.complex64,
    np.complex128 : th.complex128
}

Index = int | slice | Iterable[int]

Device = str | th.device


@dataclass
class TorchSpec:
    shape: th.Size
    dtype: th.dtype
    device: Device = "cpu"
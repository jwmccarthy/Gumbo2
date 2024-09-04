import numpy as np
import torch as th


def array_split(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]
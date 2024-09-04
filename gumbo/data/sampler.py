from itertools import chain
from collections import deque
from numpy.random import permutation

from gumbo.data.utils import array_split


class BatchSampler:

    def __init__(self, batch_size, num_epochs=1):
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def _permute(self, n):
        return array_split(permutation(n), self.batch_size)

    def _random_indices(self, n):
        return deque(chain(
            *[self._permute(n) for _ in range(self.num_epochs)]))

    def sample(self, data):
        indices = self._random_indices(len(data))
        while len(indices) > 0:
            yield data[indices.popleft()]
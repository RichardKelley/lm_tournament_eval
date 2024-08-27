
import random
import abc
from typing import List


class Scheduler(abc.ABC):
    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self
    
    @abc.abstractmethod
    def __next__(self):
        pass

class FileScheduler(Scheduler):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.idxs = []
        self.current_idx = 0

        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_idxs = line.split(',')
                line_idxs = [int(idx) for idx in line_idxs]
                self.idxs.append(line_idxs)

    def __next__(self):
        if self.current_idx >= len(self.idxs):
            raise StopIteration
        else:
            next_idxs = self.idxs[self.current_idx]
            self.current_idx += 1
            return next_idxs

class SamplingScheduler(Scheduler):
    def __init__(self, n_idxs, sample_size):
        self.n_idxs = n_idxs

        assert(sample_size > 0)
        self.sample_size = sample_size

    def __next__(self):
        return random.sample(range(self.n_idxs), self.sample_size)
        

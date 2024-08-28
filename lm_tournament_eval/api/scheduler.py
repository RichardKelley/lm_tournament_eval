
import random
import abc
from typing import List
import logging

class Scheduler(abc.ABC):
    def __init__(self) -> None:
        pass

    def __iter__(self):
        return self
    
    def set_task_size(self, n):
        self.task_size = n        
    
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
    def __init__(self, rounds : int, sample_size : int, n = None):
        '''
        A SamplingScheduler uses random sampling with replacement to generate
        a set of indices for a match between two models.

        We allow n to be None because we may reuse a SamplingScheduler across 
        multiple tasks, which will generally have different sizes. We expect 
        the user of the SamplingScheduler to set a new value of n whenever 
        the user switches to a new Task. This is not enforced in this code.

        Args:
            rounds (int):
                The number of rounds to go. 
            sample_size (int):
                The requested sample size. May be revised down if the requested sample
                size is greater than the number of possible instances.
            n (int):
                The total number of possible instances.
        '''
        assert(rounds > 0)
        self.rounds = rounds
        self.current_round = 0

        if n is not None:
            assert(n > 0)
        self.task_size = n

        self.requested_sample_size = sample_size

        assert(self.requested_sample_size > 0)
        if self.task_size is not None:
            self.sample_size = min(self.requested_sample_size, self.task_size)
        else:
            self.sample_size = sample_size

    def set_task_size(self, n : int):
        assert(n > 0)
        self.task_size = n

        if self.sample_size > self.task_size:
            logging.warn(f"Revising sample size from {self.sample_size} to {self.task_size} to handle smaller limit.")
            self.sample_size = self.task_size

    def __next__(self):
        if self.current_round > self.rounds:
            raise StopIteration
        else:
            next_match = random.sample(range(self.task_size), self.sample_size)
            self.current_round += 1
            return next_match
        


import random
import abc
from typing import List
import logging

class Scheduler(abc.ABC):
    def __init__(self) -> None:
        self.instances_seen = 0
        self.max_instances = float('inf')

        self.match_size = 0
        self.task_size = float('inf')
        self.limit = float('inf')

    def reset(self):
        self.match_size = self.original_match_size
        self.instances_seen = 0

    def __iter__(self):
        return self
    
    def set_task_size(self, n):
        self.task_size = n
        if self.task_size < self.match_size:
            self.match_size = self.task_size

        self.reset()

    def set_match_size(self, new_match_size):
        assert(new_match_size > 0)

        if self.limit < new_match_size:
            self.match_size = self.limit
        else:
            self.match_size = new_match_size

    def set_limit(self, new_limit):
        assert(new_limit > 0)
        self.limit = new_limit
        if self.limit < self.match_size:
            self.match_size = self.limit
    
    @abc.abstractmethod
    def __next__(self):
        pass

class DefaultScheduler(Scheduler):
    def __init__(self, rounds : int, match_size : int):
        super().__init__()

        assert(rounds > 0)
        self.rounds = rounds

        assert(match_size > 0)
        self.original_match_size = match_size
        self.match_size = match_size

        self.max_instances = self.rounds * self.match_size        


    def __next__(self):
        if self.instances_seen >= self.limit or self.instances_seen >= self.max_instances:
            raise StopIteration
        
        if self.instances_seen + self.match_size > self.limit:
            self.set_match_size(self.limit - self.instances_seen)
            
        idxs = list(range(self.instances_seen, self.instances_seen+self.match_size))
        self.instances_seen += len(idxs)
        return idxs

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
    def __init__(self, rounds : int, match_size : int):
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
            match_size (int):
                The requested sample size. May be revised down if the requested sample
                size is greater than the number of possible instances.
            n (int):
                The total number of possible instances.
        '''
        super().__init__()

        assert(rounds > 0)
        self.rounds = rounds

        assert(match_size > 0)
        self.original_match_size = match_size
        self.set_match_size(match_size)

        self.max_instances = self.rounds * self.match_size

    def __next__(self):
        if self.instances_seen >= self.limit or self.instances_seen >= self.max_instances:
            raise StopIteration
        
        if self.instances_seen + self.match_size > self.limit:
            self.set_match_size(self.limit - self.instances_seen)

        idxs = random.sample(range(self.task_size), self.match_size)
            
        self.instances_seen += len(idxs)
            
        return idxs

        

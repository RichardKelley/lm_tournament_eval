# An offline match scheduler
from dataclasses import dataclass, field
import numpy as np

@dataclass
class OfflineMatchSchedulerConfig:
    rounds : int
    match_size : int

class OfflineMatchScheduler:
    def __init__(self, config : OfflineMatchSchedulerConfig):
        self.config = config

    def schedule_tournament(self):
        print("Creating Offline Schedule")
        match_indices = np.random.choice(self.config.match_size,
        			                     self.config.rounds,
                                         replace=False)
        return match_indices

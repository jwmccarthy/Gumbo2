import numpy as np
from tqdm import tqdm

from gumbo.data.bundle import ListBundle


class Logger:

    def __init__(self):
        self.episodic_log = ListBundle()
        self.training_log = ListBundle()

    def start(self, num_steps):
        self.progress = tqdm(total=num_steps, leave=0)

    def log_episodic(self, data):
        for e in data.episodes:
            self.episodic_log.append(
                rew=e.rew.sum().item(), len=len(e))
            
        # mean episodic stats
        self.training_log.append(
            rew=np.mean(self.episodic_log.rew[-100:]),
            len=np.mean(self.episodic_log.len[-100:])
        )

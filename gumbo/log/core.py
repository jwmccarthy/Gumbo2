import sys

import numpy as np
from tqdm import tqdm

from gumbo.data.bundle import ListBundle


class Logger:

    def __init__(self):
        self.episodic_log = ListBundle()
        self.training_log = ListBundle()

    def start(self, num_steps):
        self.progress = tqdm(total=num_steps, file=sys.stdout)

    def stop(self):
        self.progress.close()

    def log_episodic(self, data):
        for e in data.episodes:
            self.episodic_log.append(dict(
                rew=e.rew.sum().item(), len=len(e)))
            
        # rolling episodic stats
        self.training_log.append(dict(
            rew=np.mean(self.episodic_log.rew[-100:]),
            std=np.std(self.episodic_log.rew[-100:]),
            len=np.mean(self.episodic_log.len[-100:])
        ))

    def log_training(self, data):
        self.training_log.append(data)

    def set_progress(self, steps):
        self.progress.update(steps)
        self.progress.refresh()
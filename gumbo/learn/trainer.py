import numpy as np

from gumbo.log import Logger


class Trainer:

    def __init__(self, collector, algorithm, logger=None):
        self.collector = collector
        self.algorithm = algorithm

    def train(self, num_steps):
        global_t = 0

        #self.logger.start(num_steps)

        ep_rew = []

        for data in self.collector.collect(num_steps):
            info = self.algorithm.update(data)
            
            # log episode, update info
            # if self.logger:
            #     self.logger.log_episodic(data)
            #     self.logger.log_training(info)
            #     self.logger.set_progress(global_t)

            global_t += len(data)

            for e in data.episodes:
                ep_rew.append(e.rew.sum().item())

            print(f"Episode reward: {np.mean(ep_rew[-100:])}")

            if info.get("stop"): break
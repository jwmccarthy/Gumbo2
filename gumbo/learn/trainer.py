import numpy as np

from gumbo.log import Logger


class Trainer:

    def __init__(self, collector, algorithm, logger=None):
        self.collector = collector
        self.algorithm = algorithm

    def train(self, num_steps):
        global_t = 0

        #self.logger.start(num_steps)

        for data in self.collector.collect(num_steps):
            info = self.algorithm.update(data)
            
            # log episode, update info
            # if self.logger:
            #     self.logger.log_episodic(data)
            #     self.logger.log_training(info)
            #     self.logger.set_progress(global_t)

            global_t += len(data)

            list_ep_rew, list_ep_len = [], []
            mean_ep_rew, mean_ep_len = 0, 0
            for e in data.episodes:
                mean_ep_rew += e.rew.sum().item()
                mean_ep_len += len(e)
            list_ep_rew.append(mean_ep_rew / len(data.episodes))
            list_ep_len.append(mean_ep_len / len(data.episodes))
            rolling_ep_rew = np.mean(list_ep_rew[-200:])
            rolling_ep_len = np.mean(list_ep_len[-200:])
            print(f"mean_ep_rew: {rolling_ep_rew}, mean_ep_len: {rolling_ep_len}")

            print(info)

            if info.get("stop"): break
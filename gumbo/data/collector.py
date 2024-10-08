import torch as th

from gumbo.env.gym import TorchEnv

from gumbo.data.buffer import Buffer
from gumbo.data.buffer import EpisodicBuffer

from gumbo.modules.policy import BasePolicy


# TODO: EpisodicCollector?
# TODO: ParallelCollector?
# TODO: Handle on-policy & off-policy
class Collector:
    """
    Collects data from an environment via the given policy
    """

    def __init__(
        self, 
        env: TorchEnv, 
        policy: BasePolicy, 
        buffer: Buffer,
    ):
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.device = buffer.device

    @th.no_grad()
    def _fill_buffer(self):
        self.buffer.clear()

        obs = self.env.reset()

        for t in range(self.length):
            stop = (t == self.length - 1)

            # collect experience
            act = self.policy(obs)
            next_obs, rew, info = self.env.step(
                act, truncate=stop)
            
            # store experience
            self.buffer[t] = dict(
                obs=obs, act=act, rew=rew)
            
            # make episode on end
            if "final_obs" in info:
                length = info.pop("ep_length")
                info["idx"] = slice(t + 1 - length, t + 1)
                self.buffer.add_episode(info)

            obs = next_obs

        return self.buffer.copy()
    
    def collect(self, steps: int):
        global_t = 0
        while global_t < steps:
            global_t += self.length
            yield self._fill_buffer()
        
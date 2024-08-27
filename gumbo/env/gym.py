import torch as th

from gumbo.types import TorchSpec
from gumbo.env.spaces import _torch_space


class TorchEnv:

    def __init__(self, env_fn, device="cpu", **env_kwargs):
        self.env = env_fn(**env_kwargs)
        self.device = device

        # env input/output specs
        self.obs_spec = _torch_space(
            self.env.observation_space, device=self.device)
        self.act_spec = _torch_space(
            self.env.action_space, device=self.device)
        self.rew_spec = TorchSpec(
            (), th.float32, device=self.device)
        
        # env spec for data initialization
        self.env_spec = dict(obs=self.obs_spec, 
                             act=self.act_spec, 
                             rew=self.rew_spec
        )

        self.length = 0  # current episode length

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.length = 0
        obs, _ = self.env.reset()
        return th.as_tensor(obs, device=self.device)

    def step(self, act, truncate=False):
        if isinstance(act, th.Tensor):
            act = act.cpu().numpy()

        *exp, info = self.env.step(act)

        obs, rew, trm, trc = [
            th.as_tensor(d, device=self.device) for d in exp]
        
        self.length += 1
        
        # may manually truncate
        if trm or trc or truncate:
            info = dict(
                final_obs=obs,
                truncated=trc | truncate,
                ep_length=self.length,
                **info
            )
            obs = self.reset()

        return obs, rew, info
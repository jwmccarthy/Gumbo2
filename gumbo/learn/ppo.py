import torch as th
import torch.nn.functional as F

from gumbo.learn.utils import normalize
from gumbo.learn.utils import compute_advantages_and_returns


class PPO:

    def __init__(
        self, 
        policy, 
        critic,
        sampler,
        optimizer,
        lmbda=0.95,
        gamma=0.99,
        epsilon=0.2,
        val_coef=0.5,
        ent_coef=0.0
    ):
        self.policy = policy
        self.critic = critic
        self.sampler = sampler
        self.optimizer = optimizer

        # hyperparameters
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.val_coef = val_coef
        self.ent_coef = ent_coef

        # initialize optimizer
        self.optimizer.build(policy, critic)
    
    @th.no_grad()
    def setup(self, data):
        # action log probabilities
        data.set(lgp=self.policy.log_probs(data.obs, data.act))

        # value estimates
        data.set(val=self.critic(data.obs))
        for e in data.episodes:
            e.final_val = e.truncated * self.critic(e.final_obs)

        # advantages & returns
        compute_advantages_and_returns(data, self.lmbda, self.gamma)

        return data
    
    def loss(self, data):
        val = self.critic(data.obs)

        dist = self.policy.dist(data.obs)
        lgp = dist.log_prob(data.act)
        ent = dist.entropy()

        with th.no_grad():
            approx_kl = (lgp - data.lgp).mean().item()

        # policy loss
        norm_adv = normalize(data.adv)
        ratios = th.exp(lgp - data.lgp)
        p_loss = -th.min(
            norm_adv * ratios, 
            norm_adv * th.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
        ).mean()

        # value loss
        v_loss = self.val_coef * F.mse_loss(data.ret, val)

        # entropy loss
        e_loss = self.ent_coef * -ent.mean()

        # total loss
        t_loss = p_loss + v_loss + e_loss

        return t_loss, dict(
            policy_loss=p_loss.item(),
            critic_loss=v_loss.item(),
            entropy_loss=e_loss.item(),
            approx_kl=approx_kl
        )
    
    def update(self, data):
        data = self.setup(data)
        for b in self.sampler.sample(data):
            loss, info = self.loss(b)
            self.optimizer.update(loss)
        return info
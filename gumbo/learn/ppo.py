import torch as th

from gumbo.learn.utils import compute_advantages


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
        ent_coef=0.0,
        num_epochs=10,
        batch_size=64
    ):
        self.policy = policy
        self.critic = critic
        self.sampler = sampler
        self.optimizer = optimizer

        # train dimensions
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # hyperparameters
        self.lmbda = lmbda
        self.gamma = gamma
        self.epsilon = epsilon
        self.val_coef = val_coef
        self.ent_coef = ent_coef

        # initialize optimizer
    
    @th.no_grad()
    def setup(self, data):
        # action log probabilities
        data.set(lgp=self.policy.log_probs(data.obs, data.act))

        # value estimates
        data.set(val=self.critic(data.obs))
        for e in data.episodes:
            e.final_val = self.critic(e.final_obs)

        # advantages and returns
        data = compute_advantages(data, self.lmbda, self.gamma)

        return data
    
    def loss(self, data):
        ...

    def update(self, data):
        ...
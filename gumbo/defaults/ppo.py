from torch.optim import Adam

from gumbo.data.buffer import EpisodicBuffer
from gumbo.data.collector import Collector
from gumbo.data.sampler import BatchSampler

from gumbo.modules.encoder import FlattenEncoder
from gumbo.modules.core import MLP
from gumbo.modules.operator import ValueOperator

from gumbo.optimizer import Optimizer

from gumbo.defaults.core import policy_from_env

from gumbo.learn import PPO
from gumbo.learn import Trainer


BUFFER_SIZE = 2048
BATCH_SIZE = 64
NUM_EPOCHS = 10


def default_ppo(env):
    buffer = EpisodicBuffer.from_spec(
        env.env_spec, BUFFER_SIZE)

    policy = policy_from_env(env)(
        encoder=FlattenEncoder(),
        body=MLP()
    ).build(env.obs_spec, env.act_spec)

    critic = ValueOperator(
        encoder=FlattenEncoder(),
        body=MLP()
    ).build(env.obs_spec)

    collector = Collector(env, policy, buffer)

    sampler = BatchSampler(BATCH_SIZE, NUM_EPOCHS)

    optimizer = Optimizer(Adam)

    ppo = PPO(policy, critic, sampler, optimizer)

    return Trainer(collector, ppo)
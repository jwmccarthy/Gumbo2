import torch as th
import torch.nn.functional as F

from torchaudio.functional import fftconvolve


def _compute_episode_advantages(episode, lmbda=0.95, gamma=0.99):
    next_vals = th.cat((episode.val[1:], episode.final_val))
    td_errors = episode.rew + gamma * next_vals - episode.val

    T = len(episode)

    # kernel of discount geometric series
    kernel = (gamma * lmbda) ** th.arange(
        T, dtype=th.float32).flip(0).to(episode.device)
    
    episode.adv[:] = fftconvolve(td_errors, kernel)[-T:]


def compute_advantages_and_returns(data, lmbda=0.95, gamma=0.99):
    assert "val" in data
    assert "rew" in data

    data.set(adv=th.zeros_like(data.rew))

    for e in data.episodes:
        _compute_episode_advantages(e, lmbda, gamma)

    data.set(ret=data.adv + data.val)


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)
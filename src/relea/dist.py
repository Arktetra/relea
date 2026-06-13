from torch.distributions import Distribution

import torch
import torch.distributions as D

class Prob2D(Distribution):
  def __init__(self, loc, scale):
    assert loc.ndim == 1 and loc.shape[-1] == 2, "loc must be 2D vector."
    assert scale.ndim == 1 and scale.shape[-1] == 2, "scale must be 2D vector."
    self.loc = loc
    self.scale = scale
    self.dist1 = D.normal.Normal(loc=loc, scale=scale)
    self.dist2 = D.normal.Normal(loc=loc + torch.tensor([0, 3]), scale=scale)
    self.dist3 = D.normal.Normal(loc=loc + torch.tensor([0, -3]), scale=scale)
    self.dist4 = D.normal.Normal(loc=loc + torch.tensor([3, 0]), scale=scale)
    self.dist5 = D.normal.Normal(loc=loc + torch.tensor([-3, 0]), scale=scale)

  def sample(self, N):
    n = int(N // 5)
    samples = []
    samples.append(self.dist1.sample((n,)))
    samples.append(self.dist2.sample((n,)))
    samples.append(self.dist3.sample((n,)))
    samples.append(self.dist4.sample((n,)))
    samples.append(self.dist5.sample((n,)))
    return torch.concat(samples)

  def log_prob(self, val):
    sum_ = 0
    sum_ += self.dist1.log_prob(val).sum(dim=-1).exp()
    sum_ += self.dist2.log_prob(val).sum(dim=-1).exp()
    sum_ += self.dist3.log_prob(val).sum(dim=-1).exp()
    sum_ += self.dist4.log_prob(val).sum(dim=-1).exp()
    sum_ += self.dist5.log_prob(val).sum(dim=-1).exp()
    return sum_ / 5
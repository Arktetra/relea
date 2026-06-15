from relea.samplers import Sampler

import torch

class FlowEulerSampler(Sampler):
    def __init__(self, flow_model):
        super().__init__(model=flow_model)

    def sample(self, x, steps: int):
        with torch.no_grad():
            h = 1 / steps
            xs = [x.clone()]
            for i in range(steps):
                xs.append(self.model(x, torch.full((x.size(0), 1), i * h)))
                x += xs[-1] * h
            
        return x, xs
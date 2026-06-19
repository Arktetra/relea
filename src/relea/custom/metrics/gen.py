"""
Custom metrics.
"""

from copy import copy, deepcopy
from torcheval.metrics import (
    FrechetInceptionDistance,
    Mean
)

from relea.callbacks.metrics import MetricsCallback
from relea.utils.general_utils import to_cpu

import relea
import torch

class ImageGenMetricsCallback(MetricsCallback):
    """
    Metrics callback for image generative models.

    Expects the Module wrapping the model to have a sample method for 
    sampling images.
    """
    def __init__(
        self, 
        num_samples: int = 4096,
        num_steps: int = 12,
        verbose: bool = False, 
        device: str = "cpu"
    ):
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.verbose = verbose
        self.device = device
        self.all_metrics = {}
        self.all_metrics["train_fid"] = self.train_fid = FrechetInceptionDistance(device=device)
        self.all_metrics["train_loss"] = self.train_loss = Mean()

    def after_batch(self, trainer: "relea.IterativeTrainer"):
        X, y = trainer.batch

        if trainer.training:
            for key in self.all_metrics:
                if "loss" in key:
                    self.all_metrics[key].update(to_cpu(trainer.loss))

    def after_eval(self, trainer: "relea.IterativeTrainer"):
        log = {}
        log["step"] = trainer.step 

        n = 0
        while n < self.num_samples:
            X, _ = next(trainer.dataloader)
            self.train_fid.update(X, is_real=True)
            self.train_fid.update(
                trainer.model.sample(torch.randn_like(X, device=self.device), num_steps=self.num_steps), 
                is_real=False
            )
            n += len(X)

        for k, v in self.all_metrics.items():
            log.update({k: f"{v.compute()}"})

        if self.verbose:
            self._log(log)

    def before_train(self, trainer: "relea.IterativeTrainer"):
        for key in trainer.model.__dict__.keys():
            if "loss" in key:
                self.all_metrics[key] = Mean()
        
        trainer.metrics = self
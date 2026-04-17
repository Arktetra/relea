from relea.callbacks import Callback
from relea.utils.general_utils import to_cpu
from copy import copy, deepcopy
from torcheval.metrics import Mean

import relea
import torch


class MetricsCallback(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[type(o).__name__] = o

        self.metrics = metrics

        self.train_metrics = {}
        self.val_metrics = {}
        for key, value in self.metrics.items():
            self.train_metrics[f"train_{key}"] = value
            self.val_metrics[f"val_{key}"] = deepcopy(value)

        self.all_metrics = copy(self.train_metrics)
        self.all_metrics.update(copy(self.val_metrics))
        self.all_metrics["train_loss"] = self.train_loss = Mean()
        self.all_metrics["val_loss"] = self.val_loss = Mean()

    def _log(self, log_dict):
        # print(log_dict)
        for k, v in log_dict.items():
            if k == "epoch":
                print(f"{k} - {v}")
            else:
                print(f"    {k} - {v}")

    def before_train(self, trainer: "relea.Trainer"):
        trainer.metrics = self  # type: ignore

    def before_epoch(self, trainer: "relea.Trainer"):
        [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, trainer: "relea.Trainer"):
        log = {}
        log["epoch"] = trainer.epoch

        for k, v in self.all_metrics.items():
            if isinstance(v.compute(), torch.Tensor):
                log.update({k: f"{v.compute()}"})
            else:
                log.update({k: f"{v.compute():.4f}"})

        self._log(log)

    def after_batch(self, trainer: "relea.Trainer"):
        y = to_cpu(trainer.batch)

        if trainer.training:
            for m in self.train_metrics.values():
               m.update(to_cpu(trainer.preds), y)
            
            self.train_loss.update(to_cpu(trainer.loss))  # type: ignore
        else:
            for m in self.val_metrics.values():
                m.update(to_cpu(trainer.preds))
            
            self.val_loss.update(to_cpu(trainer.loss))  # type: ignore

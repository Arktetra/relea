from relea.utils.general_utils import to_cpu
from relea.callbacks import Callback
from copy import copy, deepcopy
from torcheval.metrics import Mean
from torcheval.metrics import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio

import relea
import torch

class MetricsCallback(Callback):
    def __init__(self, val: bool = True, verbose = False, *ms, **metrics):
        self.val = val
        for o in ms:
            metrics[type(o).__name__] = o

        self.metrics = metrics
        self.verbose = verbose

        self.train_metrics = {}
        for key, value in self.metrics.items():
            self.train_metrics[f"train_{key}"] = value
        self.all_metrics = copy(self.train_metrics)
        self.all_metrics["train_loss"] = self.train_loss = Mean()

        if val:
            self.val_metrics = {}
            for key, value in self.metrics.items():
                self.val_metrics[f"val_{key}"] = deepcopy(value)
            self.all_metrics.update(copy(self.val_metrics))
            self.all_metrics["val_loss"] = self.val_loss = Mean()

    def _log(self, log_dict):
        for k, v in log_dict.items():
            if k == "epoch" or k == "step":
                print(f"{k} - {v}")
            else:
                print(f"    {k} - {v}")

    def before_train(self, trainer: "relea.Trainer"):
        trainer.metrics = self  # type: ignore

    def before_eval(self, trainer: "relea.IterativeTrainer"):
        [o.reset() for o in self.all_metrics.values()]

    def after_eval(self, trainer: "relea.IterativeTrainer"):
        log = {}
        log["step"] = trainer.step
        for k, v in self.all_metrics.items():
            log.update({k: f"{v.compute()}"})
        
        if self.verbose:
            self._log(log)

    def before_epoch(self, trainer: "relea.EpochalTrainer"):
        [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, trainer: "relea.EpochalTrainer"):
        log = {}
        log["epoch"] = trainer.epoch

        for k, v in self.all_metrics.items():
            if isinstance(v.compute(), torch.Tensor):
                log.update({k: f"{v.compute()}"})
            else:
                log.update({k: f"{v.compute():.4f}"})

        if self.verbose:
            self._log(log)

    def after_batch(self, trainer: "relea.Trainer"):
        y = to_cpu(trainer.batch[-1])

        if trainer.training:
            for m in self.train_metrics.values():
                m.update(to_cpu(trainer.preds), y)
            
            self.train_loss.update(to_cpu(trainer.loss))  # type: ignore
        else:
            if self.val:
                for m in self.val_metrics.values():
                    m.update(to_cpu(trainer.preds), y)
                
                self.val_loss.update(to_cpu(trainer.loss))  # type: ignores

class VAEMetricsCallback(MetricsCallback):
    def __init__(self, *ms, **metrics):
        device = metrics.pop("device")
        super().__init__(*ms, **metrics)
        # self.all_metrics["train_fid_score"] = self.train_fid_score = FrechetInceptionDistance(device=device)
        # self.all_metrics["val_fid_score"] = self.val_fid_score = FrechetInceptionDistance(device=device)
        # self.all_metrics["train_lpips_score"] = self.train_lpips_score = LearnedPerceptualImagePatchSimilarity()
        # self.all_metrics["val_lpips_score"] = self.val_lpips_score = LearnedPerceptualImagePatchSimilarity()
        self.all_metrics["train_psnr"] = self.train_psnr = PeakSignalNoiseRatio((0, 1))
        self.all_metrics["val_psnr"] = self.val_psnr = PeakSignalNoiseRatio((0, 1))
        self.all_metrics["train_loss_recon"] = self.train_loss_recon = Mean()
        self.all_metrics["train_loss_reg"] = self.train_loss_reg = Mean()
        self.all_metrics["val_loss_recon"] = self.val_loss_recon = Mean()
        self.all_metrics["val_loss_reg"] = self.val_loss_reg = Mean()

    def after_batch(self, trainer: "relea.VAETrainer"):
        X, y = trainer.batch

        if trainer.training:
            for m in self.train_metrics.values():
               m.update(to_cpu(trainer.preds), y)
            
            # self.train_fid_score.update(to_cpu(trainer.preds), is_real=False)
            # self.train_fid_score.update(to_cpu(torch.sigmoid(X)), is_real=True)
            # self.train_lpips_score.update(to_cpu(torch.tanh(trainer.preds)), to_cpu(torch.tanh(X)))
            self.train_psnr.update(to_cpu(torch.sigmoid(trainer.preds)), to_cpu(X))
            self.train_loss.update(to_cpu(trainer.total_loss))  # type: ignore
            self.train_loss_recon.update(to_cpu(trainer.loss_recon))
            self.train_loss_reg.update(to_cpu(trainer.loss_reg))
        else:
            for m in self.val_metrics.values():
                m.update(to_cpu(trainer.preds))
            
            # self.val_fid_score.update(to_cpu(trainer.preds), is_real=False)
            # self.val_fid_score.update(to_cpu(torch.sigmoid(X)), is_real=True)
            # self.val_lpips_score.update(to_cpu(torch.sigmoid(trainer)), to_cpu(torch.tanh(X)))
            self.val_psnr.update(to_cpu(torch.sigmoid(trainer.preds)), to_cpu(X))
            self.val_loss.update(to_cpu(trainer.total_loss))  # type: ignore
            self.val_loss_recon.update(to_cpu(trainer.loss_recon))
            self.val_loss_reg.update(to_cpu(trainer.loss_reg))



class CFMMetricsCallback(MetricsCallback):
    def __init__(self, *ms, **metrics):
        device = metrics.pop("device")
        super().__init__(*ms, **metrics)
        self.all_metrics["train_fid"] = self.train_fid = FrechetInceptionDistance(device=device)
        self.all_metrics["val_fid"] = self.val_fid = FrechetInceptionDistance(device=device)

    def after_batch(self, trainer: "relea.FlowMatchingTrainer"):
        X, y = trainer.batch
        self.train_loss.update(trainer.loss)

        if (trainer.step + 1) % trainer.eval_every != 0:
            return
        
        samples = trainer.sampler.sample(n=len(X), steps=12)

        self.train_fid.update(X, is_real=True)
        self.train_fid.update(samples, is_real=False)
        self.train_fid.compute()
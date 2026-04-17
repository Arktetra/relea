from pathlib import Path
from typing import Union

import relea

from relea.callbacks import Callback


class ModelCheckpoint(Callback):
    def __init__(self, dir_path: Union[Path, str], track: str = "loss", every_epoch: bool = False, persist: bool = False):
        super().__init__()
        self.track = track
        self.best_metric = float("inf") if "loss" in track else 0.
        self.best_metric_path = None
        self.every_epoch = every_epoch
        self.persist = persist

        self.dir_path = Path(dir_path) if not isinstance(dir_path, Path) else dir_path
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def after_epoch(self, trainer: "relea.Trainer"):
        if not trainer.enable_checkpointing:
            return
        
        metric = trainer.metrics.all_metrics[f"val_{self.track}"].compute()  # type: ignore

        if self.every_epoch:
            if self.persist:
                filename = f"{type(trainer.model).__name__}-{trainer.epoch}-{self.track}-{metric:.4f}.pt"
            else:
                filename = f"{type(trainer.model).__name__}.pt"
            ckpt_path = self.dir_path / filename

            trainer.model.save_checkpoint(ckpt_path)
        else:
            filename = f"{type(trainer.model).__name__}-{trainer.epoch}-{self.track}-{metric:.4f}.pt"
            ckpt_path = self.dir_path / filename

            if "loss" in self.track:
                if metric < self.best_metric:
                    if self.best_metric_path is not None:
                        self.best_metric_path.unlink()
                    self.best_metric = metric
                    self.best_metric_path = ckpt_path

                    trainer.model.save_checkpoint(ckpt_path)
            else:
                if metric > self.best_metric:
                    if self.best_metric_path is not None:
                        self.best_metric_path.unlink()
                    self.best_metric = metric
                    self.best_metric_path = ckpt_path

                    trainer.model.save_checkpoint(ckpt_path)
from relea.callbacks import Callback
from pathlib import Path
from typing import Union

import relea
import matplotlib.pyplot as plt


class RecorderCallback(Callback):
    order = 1

    def __init__(self, savepath: Union[Path, str] = "./"):
        super().__init__()

        self.savepath = savepath if isinstance(savepath, Path) else Path(savepath)
        self.savepath = self.savepath / "losses.json"

    def before_train(self, trainer: "relea.Trainer"):
        self.losses = {"train_loss": [], "val_loss": []}

    def after_batch(self, trainer: "relea.Trainer"):
        if trainer.training:
            self.losses["train_loss"].append(trainer.metrics.train_loss.compute())  # type: ignore
        else:
            self.losses["val_loss"].append(trainer.metrics.val_loss.compute())  # type: ignore

    def plot(self):
        plt.plot(self.losses["train_loss"], label="train_loss")
        plt.plot(self.losses["val_loss"], label="val_loss")
        plt.legend()
        plt.show()
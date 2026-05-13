from pathlib import Path
from typing import List, Optional, Union

from relea.models.base import BaseModule
from relea.trainers.trainer import Trainer, TRAIN_DATALOADER, VAL_DATALOADER
from relea.callbacks import Callback, with_callbacks, run_callbacks
from relea.utils.general_utils import get_total_parameters, get_trainable_parameters

import matplotlib.pyplot as plt
import torch

class VAETrainer(Trainer):
    def __init__(
        self,
        accelerator: str = "cpu",
        max_epochs: Optional[int] = None,
        callbacks: List[Callback] = [],
        enable_checkpointing: bool = True,
        checkpoint_dir: Optional[Union[Path, str]] = None,
        clip_grad: Optional[bool] = False,
        sample_epoch: Optional[int] = None
    ):
        super().__init__(
            accelerator,
            max_epochs,
            callbacks,
            enable_checkpointing,
            checkpoint_dir,
            clip_grad,
        )

        self.sample_epoch = sample_epoch

    @with_callbacks("batch")
    def run_batch(self, batch):
        if self.training:
            preds, total_loss, loss_recon, loss_reg = self.model.run_step(batch)
            total_loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            preds, total_loss, loss_recon, loss_reg = self.model.run_step(batch)
        
        self.preds = preds
        self.total_loss = total_loss
        self.loss_recon = loss_recon
        self.loss_reg = loss_reg
        self.losses = {
            "total_loss": total_loss,
            "loss_recon": loss_recon,
            "loss_reg": loss_reg
        }

    @with_callbacks("train")
    def train(
        self,
        model: BaseModule,
        optimizer: torch.optim.Optimizer,
        train_dataloader: TRAIN_DATALOADER,
        val_dataloader: VAL_DATALOADER,
        savepath: Optional[str] = None
    ):
        self.model = model.to(self.accelerator)
        self.optimizer = optimizer

        if self.sample_epoch:
            fig, axs = plt.subplots(1, self.sample_epoch)

        print(f"Total number of parameters: {get_total_parameters(self.model)}")
        print(f"Total number of trainable parameters: {get_trainable_parameters(self.model)}")
        print(f"Starting Training on {self.model.device}")

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.train_epoch(train_dataloader, val_dataloader)

            if self.sample_epoch and savepath and  (self.epoch + 1) % self.sample_epoch == 0:
                imgs = self.model.sample(n_samples=self.sample_epoch)
                imgs = imgs.detach().cpu()
                
                for (ax, img) in zip(axs, imgs):
                    ax.imshow(img.permute(1, 2, 0))
                    ax.set_xticks([])
                    ax.set_yticks([])

                fig.savefig(f"{savepath}/epoch-{epoch}.png")

    def callback(self, method_name: str):
        run_callbacks(self.callbacks, method_name, self)


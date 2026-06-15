from pathlib import Path
from typing import List, Optional, Union

from relea.trainers.generative import GenerativeTrainer
from relea.callbacks import Callback, with_callbacks

import torch

class VAETrainer(GenerativeTrainer):
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
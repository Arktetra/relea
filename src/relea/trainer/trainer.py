from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, Union, List

import torch
import torch.nn as nn

from relea.callbacks import (
    Callback,
    with_callbacks,
    run_callbacks,
    RecorderCallback,
)
from relea.callbacks.checkpoint import ModelCheckpoint
from relea.models.base import BaseModule
from relea.utils.general_utils import has_instance

TRAIN_DATALOADER = DataLoader
VAL_DATALOADER = DataLoader


class Trainer:
    def __init__(
        self,
        accelerator: str = "cpu",
        max_epochs: Optional[int] = None,
        callbacks: List[Callback] = [],
        enable_checkpointing: bool = True,
        checkpoint_dir: Optional[Union[Path, str]] = None,
        clip_grad: Optional[bool] = False,
    ):
        self.accelerator = accelerator
        self.max_epochs = max_epochs if max_epochs is not None else 1000
        self.enable_checkpointing = enable_checkpointing
        self.clip_grad = clip_grad

        self.callbacks = callbacks

        if not has_instance(callbacks, RecorderCallback):
            self.callbacks.append(RecorderCallback())

        if not has_instance(callbacks, ModelCheckpoint) and enable_checkpointing:
            checkpoint_dir = checkpoint_dir if checkpoint_dir else "./ckpts"
            self.callbacks.append(ModelCheckpoint(dir_path=checkpoint_dir))

    @with_callbacks("batch")
    def run_batch(self, batch):
        if self.training:
            logits, loss = self.model.run_step(batch)
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            logits, loss = self.model.run_step(batch)

        self.preds = logits
        self.loss = loss

    @with_callbacks("epoch")
    def train_epoch(self, train_dataloader, val_dataloader):
        self.training = True
        self.model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            self.batch_idx, self.batch = batch_idx, batch
            self.run_batch(batch)

        self.training = False
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                self.batch_idx, self.batch = batch_idx, batch
                self.run_batch(batch)

    @with_callbacks("train")
    def train(
        self,
        model: BaseModule,
        train_dataloader: TRAIN_DATALOADER,
        val_dataloader: VAL_DATALOADER,
    ):
        self.model = model.to(self.accelerator)
        self.optimizer = self.model.configure_optimizers()

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.train_epoch(train_dataloader, val_dataloader)

    def callback(self, method_name: str):
        run_callbacks(self.callbacks, method_name, self)
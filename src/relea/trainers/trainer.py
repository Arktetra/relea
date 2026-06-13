from abc import ABC, abstractmethod
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Union, List

import torch

from relea.callbacks import (
    Callback,
    EpochalCallback,
    IterativeCallback,
    with_callbacks,
    run_callbacks,
)
from relea.callbacks.checkpoint import ModelCheckpoint
from relea.models.base import BaseModule
from relea.utils.general_utils import cycle, has_instance, get_total_parameters, get_trainable_parameters

TRAIN_DATALOADER = DataLoader
VAL_DATALOADER = DataLoader

class Trainer(ABC):
    @abstractmethod
    @with_callbacks("batch")
    def run_batch(self, batch):
        pass

    @abstractmethod
    @with_callbacks("train")
    def train(
        self, 
        model: BaseModule, 
        optimizer: torch.optim.Optimizer,
        train_dataloader: TRAIN_DATALOADER
    ):
        pass

    def callback(self, method_name: str):
        run_callbacks(self.callbacks, method_name, self)

class IterativeTrainer(Trainer):
    def __init__(
        self,
        accelerator: str = "cpu",
        num_train_steps: int = 1000000,
        num_val_steps: Optional[int] = None,
        eval_every: int = 10000,
        callbacks: List[IterativeCallback] = [],
        enable_checkpointing: bool = True,
        checkpoint_dir: Optional[Union[Path, str]] = None,
        clip_grad: Optional[bool] = False
    ):
        super().__init__()
        self.accelerator = accelerator
        self.num_train_steps = num_train_steps
        self.num_val_steps = num_val_steps
        self.eval_every = eval_every
        self.callbacks = callbacks
        self.enable_checkpointing = enable_checkpointing
        self.clip_grad = clip_grad

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

    @with_callbacks("eval")
    def run_eval(self):
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.num_val_steps):
                self.batch = next(iter(self.val_dataloader))
                self.run_batch(self.batch)
            
    @with_callbacks("train")
    def train(
        self,
        model: BaseModule,
        optimizer: torch.optim.Optimizer,
        train_dataloader: TRAIN_DATALOADER,
        val_dataloader: VAL_DATALOADER = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = cycle(train_dataloader)
        if val_dataloader:
            self.val_dataloader = cycle(val_dataloader)


        self.preamble()
        self.step = 0
        
        pbar = tqdm(total=self.num_train_steps)
        while self.step < self.num_train_steps:
            self.batch = next(iter(train_dataloader))

            self.training = True
            self.run_batch(self.batch)

            if (self.step + 1) % self.eval_every == 0 and val_dataloader and self.num_val_steps:
                pbar.update(self.eval_every)
                self.training = False
                self.run_eval()

            self.step += 1

        pbar.close()

    def preamble(self):
        print(f"Total number of parameters: {get_total_parameters(self.model)}")
        print(f"Total number of trainable parameters: {get_trainable_parameters(self.model)}")
        print(f"Starting training on {self.model.device}...")


# class Trainer:
#     def __init__(
#         self,
#         accelerator: str = "cpu",
#         max_epochs: Optional[int] = None,
#         callbacks: List[Callback] = [],
#         enable_checkpointing: bool = True,
#         checkpoint_dir: Optional[Union[Path, str]] = None,
#         clip_grad: Optional[bool] = False,
#     ):
#         self.accelerator = accelerator
#         self.max_epochs = max_epochs if max_epochs is not None else 1000
#         self.enable_checkpointing = enable_checkpointing
#         self.clip_grad = clip_grad

#         self.callbacks = callbacks

#         # if not has_instance(callbacks, RecorderCallback):
#         #     self.callbacks.append(RecorderCallback())

#         if not has_instance(callbacks, ModelCheckpoint) and enable_checkpointing:
#             checkpoint_dir = checkpoint_dir if checkpoint_dir else "./ckpts"
#             self.callbacks.append(ModelCheckpoint(dir_path=checkpoint_dir))

#     @with_callbacks("batch")
#     def run_batch(self, batch):
#         if self.training:
#             logits, loss = self.model.run_step(batch)
#             loss.backward()
#             if self.clip_grad:
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#         else:
#             logits, loss = self.model.run_step(batch)

#         self.preds = logits
#         self.loss = loss

#     @with_callbacks("epoch")
#     def train_epoch(self, train_dataloader, val_dataloader):
#         self.training = True
#         self.model.train()
#         for batch_idx, batch in enumerate(tqdm(train_dataloader)):
#             self.batch_idx, self.batch = batch_idx, batch
#             self.run_batch(batch)

#         self.training = False
#         self.model.eval()
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(tqdm(val_dataloader)):
#                 self.batch_idx, self.batch = batch_idx, batch
#                 self.run_batch(batch)

#     @with_callbacks("train")
#     def train(
#         self,
#         model: BaseModule,
#         optimizer: torch.optim.Optimizer,
#         train_dataloader: TRAIN_DATALOADER,
#         val_dataloader: VAL_DATALOADER,
#     ):
#         self.model = model.to(self.accelerator)
#         self.optimizer = optimizer

#         for epoch in range(self.max_epochs):
#             self.epoch = epoch
#             self.train_epoch(train_dataloader, val_dataloader)

#     def callback(self, method_name: str):
#         run_callbacks(self.callbacks, method_name, self)

#     def preamble(self):
#         print(f"Total number of parameters: {get_total_parameters(self.model)}")
#         print(f"Total number of trainable parameters: {get_trainable_parameters(self.model)}")
#         print(f"Starting training on {self.model.device}.")


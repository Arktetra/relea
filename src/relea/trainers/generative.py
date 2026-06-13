from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional, Union

from relea.models.base import BaseModule
from relea.samplers import Sampler
from relea.trainers.trainer import Trainer, TRAIN_DATALOADER
from relea.callbacks import Callback, with_callbacks, ModelCheckpoint
from relea.utils.general_utils import has_instance, cycle

import matplotlib.pyplot as plt
import torch

class GenerativeTrainer(Trainer):
    """
    Generator Matching Trainer.
    """
    def __init__(
        self,
        accelerator: str = "cpu",
        num_train_steps: int = 1000000,
        eval_every: int = 10000,
        callbacks: List[Callback] = [],
        enable_checkpointing: bool = True,
        checkpoint_dir: Optional[Union[Path, str]] = None,
        clip_grad: Optional[bool] = False,
        n_samples: int = 32
    ):
        self.accelerator = accelerator
        self.num_train_steps = num_train_steps
        self.eval_every = eval_every
        self.n_samples = n_samples
        self.callbacks = callbacks
        self.enable_checkpointing = enable_checkpointing
        self.clip_grad = clip_grad

        if not has_instance(callbacks, ModelCheckpoint) and enable_checkpointing:
            checkpoint_dir = checkpoint_dir if checkpoint_dir else "./ckpts"
            self.callbacks.append(ModelCheckpoint(dir_path=checkpoint_dir))

    @with_callbacks("eval")
    def _train_until_eval(self, train_dataloader):
        self.training = True
        self.model.train()
        for _ in range(self.eval_every):
            self.batch = next(iter(train_dataloader))
            self._run_batch(self.batch)
            self.step += 1
            self.pbar.update(1)
            if self.step % self.num_train_steps == 0:
                break

    @with_callbacks("train")
    def _train(
        self,
        train_dataloader: TRAIN_DATALOADER,
    ):
        self.step = 0
        self.pbar = tqdm(total=self.num_train_steps)
        while True:
            self._train_until_eval(train_dataloader)
            if self.step + 1 >= self.num_train_steps:
                break
        self.pbar.close()
        

    def train(
        self,
        model: BaseModule,
        sampler: Sampler,
        optimizer: torch.optim.Optimizer,
        train_dataloader: TRAIN_DATALOADER,
    ):
        self.model = model.to(self.accelerator)
        self.sampler = sampler
        self.optimizer = optimizer
        train_dataloader = cycle(train_dataloader)

        self.preamble()
        self._train(train_dataloader)

        


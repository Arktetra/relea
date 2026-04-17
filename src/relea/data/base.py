from abc import abstractmethod
from math import ceil
from torch.utils.data import DataLoader
from typing import Optional

import torch


class DataModule:
    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 1,
        on_gpu: bool = False,
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.on_gpu = on_gpu
        self.collate_fn = None

        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

    def __repr__(self):
        return f"{self.__class__.__name__}\n" \
        f"   train length - {ceil(len(self.train_dataset) / self.batch_size)}\n" \
        f"   val length - {ceil(len(self.val_dataset) / self.batch_size)}\n" \
        f"   test length - {ceil(len(self.test_dataset) / self.batch_size)}\n"

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    @abstractmethod
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    @abstractmethod
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )
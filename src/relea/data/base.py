from abc import abstractmethod
from math import ceil
from torch.utils.data import DataLoader, Dataset
from typing import Optional

import torch

class BaseDataset(Dataset):
    def __init__(
        self, dataset: Dataset, transform=None, target_transform=None
    ):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        datum, target = self.dataset[idx]
        
        if self.transform:
            datum = self.transform(datum)
        if self.target_transform:
            target = self.target_transform(datum)

        return datum, target


class DataModule:
    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        on_gpu: bool = False,
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.on_gpu = on_gpu
        self.collate_fn = None
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

    def __repr__(self):
        return (    # type: ignore
            f"{self.__class__.__name__}\n"
        f"   train length - {ceil(len(self.train_dataset) / self.batch_size)}\n"
        f"   val length - {ceil(len(self.val_dataset) / self.batch_size)}\n"
        f"   test length - {ceil(len(self.test_dataset) / self.batch_size)}\n"
        )

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.on_gpu,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    @abstractmethod
    def val_dataloader(self):
        return DataLoader( 
            self.val_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.on_gpu,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )

    @abstractmethod
    def test_dataloader(self):
        return DataLoader( 
            self.test_dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.on_gpu,
            collate_fn=self.collate_fn,
            generator=self.generator,
        )
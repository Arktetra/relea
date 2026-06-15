from relea.data.base import DataModule, BaseDataset
from relea.stem.image import ImageStem
from relea.stem.mnist import MiniImageNetGenStem

from pathlib import Path
from torchvision import datasets
from torch.utils.data import random_split
from typing import Optional

class MiniImageNetDataModule(DataModule):
    def __init__(
        self,
        root,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 1,
        on_gpu: bool = False,
        train_stem: Optional[ImageStem] = None,
        test_stem: Optional[ImageStem] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            batch_size,
            shuffle,
            num_workers,
            on_gpu,
            seed
        )

        self.data_dir = Path(root) / "data" / "processed" / "MiniImageNet"

        self.train_transform = train_stem
        self.test_transform = test_stem

    def prepare_data(self):
        pass

    def setup(self):
        train_dataset = datasets.MNIST(self.data_dir, train=True, download=True)
        train_dataset, val_dataset = random_split(
            train_dataset, [0.9, 0.1]
        )
        self.train_dataset = BaseDataset(train_dataset, transform=self.train_transform)
        self.val_dataset = BaseDataset(val_dataset, transform=self.test_transform)
        self.test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.test_transform)
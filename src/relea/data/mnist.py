from relea.data.base import DataModule

from pathlib import Path
from torchvision import datasets
from torch.utils.data import random_split
from typing import Optional

class MNISTDataModule(DataModule):
    def __init__(
        self,
        root,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 1,
        on_gpu: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            batch_size,
            shuffle,
            num_workers,
            on_gpu,
            seed
        )

        self.data_dir = Path(root) / "data" / "processed" / "MNIST"

    def prepare_data(self):
        pass

    def setup(self):
        self.train_dataset = datasets.MNIST(self.data_dir, train=True, download=True)
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [0.9, 0.1]
        )
        self.test_dataset = datasets.MNIST(self.data_dir, train=False, download=True)
from relea.data.base import DataModule
from relea.stem.imagenette import ImagenetteTrainStem, ImagenetteTestStem

from pathlib import Path
from torchvision import datasets
from torch.utils.data import random_split
from typing import Optional

class ImagenetteDataModule(DataModule):
    def __init__(
        self,
        root,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        on_gpu: bool = False,
        seed: Optional[int] = None,
        use_stem: bool = True,
        resize: int = 256
    ):
        super().__init__(
            batch_size,
            shuffle,
            num_workers,
            on_gpu,
            seed
        )

        self.data_dir = Path(root) / "data" / "processed" / "Imagenette"

        self.train_transform = None
        self.test_transform = None
        if use_stem:
            self.train_transform = ImagenetteTrainStem(resize)
            self.test_transform = ImagenetteTestStem(resize)

    def prepare_data(self):
        pass

    def setup(self):
        self.train_dataset = datasets.Imagenette(
            self.data_dir, 
            split="train", 
            download=True,
            transform=self.train_transform
        )

        self.test_dataset = datasets.Imagenette(
            self.data_dir, 
            split="val", 
            download=True,
            transform=self.test_transform
        )
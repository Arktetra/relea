from relea.data.base import BaseDataset, DataModule
from relea.stem.text import TextStem

from typing import Optional
from torch.utils.data import Dataset, random_split
from pathlib import Path

class FineWeb(DataModule):
    
    def __init__(
        self,
        root,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        on_gpu: bool = False,
        seed: Optional[int] = None,
        use_stem: bool = True,
        resize: int = 256
    ):
        super().__init__(
            batch_size,
            shuffle,
            num_workers,
            persistent_workers,
            prefetch_factor,
            on_gpu,
            seed
        )

        self.data_dir = Path(root) / "data" / "processed" / "fineweb"

        self.train_transform = None
        self.test_transform = None
        
        if use_stem:
            self.train_transform = MiniImageNetTrainStem(resize)
            self.test_transform = MiniImageNetTestStem(resize)

    def prepare_data(self):
        pass

    def setup(self):
        ds = "motionlabs/fineweb-ultra-mini"
        
        train_dataset = HFMiniImageNet(load_dataset(ds, split="train"))
        train_dataset, val_dataset = random_split(
            train_dataset, [0.8, 0.2], generator=self.generator
        )
        test_dataset = HFMiniImageNet(load_dataset(ds, split="test"))
        
        self.train_dataset = BaseDataset(train_dataset, transform=self.train_transform)
        self.val_dataset = BaseDataset(val_dataset, transform=self.test_transform)
        self.test_dataset = BaseDataset(test_dataset, transform=self.test_transform)

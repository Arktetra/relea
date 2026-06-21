from relea.data.base import DataModule, BaseDataset
from relea.stem.image import ImageStem
from relea.stem.miniimagenet import MiniImageNetTestStem, MiniImageNetTrainStem
from relea.stem.miniimagenet import MiniImageNetGenStem

from pathlib import Path
from torchvision import datasets
from torch.utils.data import random_split, Dataset
from typing import Optional
from datasets import load_dataset

class HFMiniImageNet(Dataset):
    def __init__(self, ds):
        self.ds = ds
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        sample = self.ds[index]
        return sample["image"], sample["label"] 
    
class MiniImageNetDataModule(DataModule):
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

        self.data_dir = Path(root) / "data" / "processed" / "miniimagenet"

        self.train_transform = None
        self.test_transform = None
        
        if use_stem:
            self.train_transform = MiniImageNetTrainStem(resize)
            self.test_transform = MiniImageNetTestStem(resize)

    def prepare_data(self):
        
        ds = "timm/mini-imagenet"
                
        self.train_dataset = HFMiniImageNet(load_dataset(ds, split="train[:80%]", cache_dir=self.data_dir))
        # train_dataset, val_dataset = random_split(
        #     train_dataset, [0.8, 0.2], generator=self.generator
        # )
        self.val_dataset = HFMiniImageNet(load_dataset(ds, split="train[80%:]", cache_dir=self.data_dir))
        self.test_dataset = HFMiniImageNet(load_dataset(ds, split="test", cache_dir=self.data_dir))
        
    def setup(self):
        self.train_dataset = BaseDataset(self.train_dataset, transform=self.train_transform)
        self.val_dataset = BaseDataset(self.val_dataset, transform=self.test_transform)
        self.test_dataset = BaseDataset(self.test_dataset, transform=self.test_transform)

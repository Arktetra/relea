from huggingface_hub import hf_hub_download
from pathlib import Path
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from typing import Optional, Union

from relea.data.base import DataModule
from relea.stem.ucf import UCFTrainStem, UCFTestStem

import pims
import tarfile
import torch

import relea.metadata.ucf as metadata

class UCFDataset(Dataset):
    def __init__(self, data_dir: Union[Path, str], split: str = "train", transforms = None, target_transforms = None):
        super().__init__()
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        self.split_dir = self.data_dir / split
        self.video_paths = list(self.split_dir.glob("**/*.avi"))
        self.transforms = transforms
        self.target_transforms = target_transforms
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        decoder = VideoDecoder(self.video_paths[idx], dimension_order="NCHW")
        target = video_path.parent.stem

        if self.transforms:
            video = self.transforms(decoder)
        if self.target_transforms:
            target = self.target_transforms(target)

        return video, torch.tensor(metadata.LABEL_TO_IDX[target])

class UCFDataModule(DataModule):
    def __init__(
        self,
        root: Union[Path, str],
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        on_gpu: bool = False,
        seed: Optional[int] = None
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
        self.root = Path(root) if isinstance(root, str) else root
        self.train_transforms = UCFTrainStem()
        self.test_transforms = UCFTestStem()
        self.target_transforms = None

    def prepare_data(self):
        self.raw_data_dir = self.root / "data/raw/ufc"
        repo_id = "sayakpaul/ucf101-subset"
        filename = "UCF101_subset.tar.gz"
        filepath = hf_hub_download(repo_id, filename, local_dir=self.raw_data_dir, repo_type="dataset")
        self.processed_data_dir = Path(filepath).parents[2] / "processed"

        if not (self.processed_data_dir / "UCF101_subset").exists():
            with tarfile.open(filepath) as f:
                f.extractall(path=self.processed_data_dir)
        self.processed_data_dir = self.processed_data_dir / "UCF101_subset"

    def setup(self):
        self.train_dataset = UCFDataset(
            self.processed_data_dir, 
            split="train", 
            transforms=self.train_transforms,
            target_transforms=self.target_transforms
        )
        self.val_dataset = UCFDataset(
            self.processed_data_dir, 
            split="val", 
            transforms=self.test_transforms,
            target_transforms=self.target_transforms
        )
        self.test_dataset = UCFDataset(
            self.processed_data_dir, 
            split="test", 
            transforms=self.test_transforms,
            target_transforms=self.target_transforms
        )

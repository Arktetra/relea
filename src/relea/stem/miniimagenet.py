from relea.stem.image import ImageStem

from torchvision.transforms import v2

import torch

class MiniImageNetTrainStem(ImageStem):
    def __init__(self, resize: int = 32):
        super().__init__()
        self.resize = resize
        self.transforms = v2.Compose([
            v2.Resize((self.resize, self.resize)),
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.465, 0.405],
                std=[0.229, 0.224, 0.225]
            )
        ])

class MiniImageNetTestStem(MiniImageNetTrainStem):
    def __init__(self, resize: int = 32):
        super().__init__(resize)

class MiniImageNetGenStem(ImageStem):
    def __init__(self, resize: int = 32):
        super().__init__()
        self.resize = resize
        self.transforms = v2.Compose([
            v2.Resize((self.resize, self.resize)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
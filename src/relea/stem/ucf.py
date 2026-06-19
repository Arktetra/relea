from relea.stem.image import ImageStem

from torchvision.transforms import v2
from torchcodec.decoders import VideoDecoder

import pims
import torch

class UniformSample(object):
    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames

    def __call__(self, x: VideoDecoder):
        # print(len(range(0, len(x), (len(x) + self.n_frames - 1) // self.n_frames)))
        return x.get_frames_at(indices=[i * len(x) // self.n_frames for i in range(self.n_frames)]).data

class UCFTrainStem(ImageStem):
    def __init__(self):
        super().__init__()
        self.transforms = v2.Compose([
            UniformSample(n_frames=16),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(size=(224, 224)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class UCFTestStem(ImageStem):
    def __init__(self):
        super().__init__()
        self.transforms = v2.Compose([
            UniformSample(n_frames=16),
            v2.Resize(size=(224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


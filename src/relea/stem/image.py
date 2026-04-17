from torchvision.transforms import v2

import torch


class ImageStem:
    """
    A stem for models operating on images.

    Images are presumed to be provided as torch tensors.
    """

    def __init__(self):
        self.transforms = v2.Compose([torch.nn.Identity()])

    def __call__(self, img):
        return self.transforms(img)
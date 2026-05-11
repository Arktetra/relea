#%%
%load_ext autoreload 
%autoreload 2   

#%%
from relea.data import ImagenetteDataModule
from relea.models import VEncoder, VDecoder
from relea.utils.general_utils import get_total_parameters

from torchvision.transforms import v2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# %%
datamodule = ImagenetteDataModule(
    "../", 
    batch_size=128, 
    shuffle=True, 
    num_workers=0, 
    on_gpu=False
)
datamodule.prepare_data()
datamodule.setup()
# %%
train_dataloader = datamodule.train_dataloader()
# %%
X, y = next(iter(train_dataloader))
# %%
X.shape
# %%
encoder = VEncoder().to("mps")
decoder = VDecoder().to("mps")
# %%
z_loc, z_scale = encoder(X.to("mps"))
# %%
z_loc.shape, z_scale.shape
# %%
x_pred = decoder(z_loc)
# %%
l = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
x = torch.randn(1, 3, 64, 64)
l(x).shape
# %%
x_pred.shape
# %%
get_total_parameters(encoder), get_total_parameters(decoder)
# %%
z_loc.shape
# %%
l1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
# %%
l1(z_loc).shape
# %%
z_loc.shape
# %%
import torch
import torch.nn as nn
# %%
hidden_channels = [8, 16, 32]
modules = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3), 
        nn.AvgPool2d(3),
        nn.ReLU(),
    )
    for (in_channels, out_channels) in zip(hidden_channels[:-1], hidden_channels[1:])
])
# %%
modules
# %%
nn.Sequential(*modules)
# %%
nn.ModuleList()
[1] + [1, 2]
# %%
2 * 2 * 512
# %%

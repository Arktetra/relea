#%%
%load_ext autoreload 
%autoreload 2   

#%%
from relea.data import ImagenetteDataModule
from relea.data.mnist import MNISTDataModule
from relea.models import VEncoder, VDecoder
from relea.utils.general_utils import get_total_parameters

from torchvision.transforms import v2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# %%
datamodule = MNISTDataModule(
    "../", 
    batch_size=256, 
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
from relea.models import VAE
from tqdm import tqdm

import yaml
# %%
with open("../configs/vae/mnist.yaml") as stream:
    cfg = yaml.safe_load(stream)
# %%
vae = VAE.from_config(cfg).to("mps")
# %%
vae.load_checkpoint("../ckpts/VAE.pt")
# %%
batch = next(iter(train_dataloader))
#%%
vae.sample(1)
#%%
imgs, labels = batch
vae.encoder(torch.randn((256, 3, 256)))
#%%
ncols = 5
fig, axs = plt.subplots(1, ncols)
preds, _, _, _ = vae.run_step(batch)
print(preds.shape)

for ax, img in zip(axs, preds[:ncols]):
    ax.imshow(img.cpu().detach().permute(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])
# %%
img = vae.sample(1)
# %%
plt.imshow(
    torch.sigmoid(
        img.squeeze().detach().cpu().permute(1, 2, 0)
    )
)
# %%
optim = torch.optim.Adam(vae.parameters(), lr=1e-4)
for batch in tqdm((train_dataloader)):
    X_pred, total_loss, loss_recon, loss_reg  = vae.run_step(batch)
    total_loss.backward()
    optim.step()
    optim.zero_grad()
# %%

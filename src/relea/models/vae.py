from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from relea.models.base import BaseModule

class VEncoder(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        latent_dim=128,
        hidden_channels=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
    
        if hidden_channels is None:
            hidden_channels = [8, 16, 32, 64, 128]
        
        self.hidden_channels = [in_channels] + hidden_channels

        modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=2),
                nn.Softplus(),
                nn.Dropout()
            )
            for (in_channels, out_channels) in zip(self.hidden_channels[:-1], self.hidden_channels[1:])
        ])

        self.conv_blocks = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_loc = nn.Linear(hidden_channels[-1], self.latent_dim)
        self.fc_scale = nn.Linear(hidden_channels[-1], self.latent_dim)

    def forward(self, x):
        z = self.conv_blocks(x)
        z = self.gap(z).squeeze()
        z_loc = self.fc_loc(z)
        z_scale = self.fc_scale(z)
        return z_loc, z_scale

class VDecoder(nn.Module):
    def __init__(
        self, 
        latent_dim=128,
        out_channels=3,
        hidden_channels: List[int] = None, # the length of this list affects the spatial size.
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [128, 64, 32, 16, 8]

        if len(hidden_channels) < 7:
            hidden_channels = hidden_channels + (7 - len(hidden_channels)) * [out_channels]

        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(latent_dim, hidden_channels[0] * 2 * 2)

        modules = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.Softplus()
            )
            for (in_channels, out_channels) in zip(self.hidden_channels[:-1], self.hidden_channels[1:])
        ])

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.Tanh()
            )
        )

        self.conv_blocks = nn.Sequential(*modules)

    def forward(self, z):
        x = self.fc(z).view(-1, self.hidden_channels[0], 2, 2)
        x = self.conv_blocks(x)
        return x
    
class VAE(BaseModule):
    def __init__(
        self, 
        encoder: nn.Module,
        decoder: nn.Module,
        weight_recon: float = 1,
        weight_reg: float = 1,  # KL term - latent space regularization
        device: str = "cpu"
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.weight_recon = weight_recon
        self.weight_reg = weight_reg
        self.device = device

    def encode(self, x):
        z_loc, z_scale = self.encoder(x)
        return z_loc, z_scale
    
    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, z_loc, z_scale):
        epsilon = torch.randn_like(z_loc)
        return z_scale * epsilon + z_loc

    def run_step(self, batch):
        X = batch

        z_loc, z_scale = self.encode(X)
        z = self.reparameterize(z_loc, z_scale)
        X_pred = self.decode(z)

        loss_recon = F.mse_loss(X, X_pred)
        loss_reg = - 0.5 * torch.mean(
            torch.sum(
                1 + torch.log(z_scale ** 2) - z_scale ** 2 - z_loc ** 2,
                dim=z.shape[1:]
            ),
            dim=0
        )

        return self.weight_recon * loss_recon + self.weight_reg * loss_reg
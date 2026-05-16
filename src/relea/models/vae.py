from typing import List

import sys
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
        self.fc_mu = nn.Linear(hidden_channels[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels[-1], self.latent_dim)

    def forward(self, x):
        z = self.conv_blocks(x)
        z = self.gap(z).squeeze()
        z_mu = self.fc_mu(z)
        z_logvar = self.fc_logvar(z)
        return z_mu, z_logvar

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

        # if len(hidden_channels) < 7:
        #     hidden_channels = hidden_channels + (7 - len(hidden_channels)) * [out_channels]

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
                nn.ConvTranspose2d(self.hidden_channels[-1], out_channels, kernel_size=2, stride=2, padding=0),
                # nn.BatchNorm2d(out_channels),
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

    @staticmethod
    def from_config(cfg):
        modules = sys.modules[__name__]
        encoder = getattr(modules, cfg["models"]["args"]["encoder"]["name"])(
            **cfg["models"]["args"]["encoder"]["args"]
        )
        decoder = getattr(modules, cfg["models"]["args"]["decoder"]["name"])(
            **cfg["models"]["args"]["decoder"]["args"]
        )

        return VAE(
            encoder,
            decoder,
            weight_recon = cfg["models"]["args"]["weight_recon"],
            weight_reg = cfg["models"]["args"]["weight_reg"],
            device=cfg["trainer"]["args"]["device"]
        )


    def encode(self, x):
        z_mu, z_logvar = self.encoder(x)
        return z_mu, z_logvar
    
    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(z_mu)
        return z_std * epsilon + z_mu
    
    def sample(self, n_samples):
        z = torch.randn((n_samples, self.encoder.latent_dim)).to(self.device)
        return self.decode(z)

    def run_step(self, batch):
        X, _ = batch
        X = X.to(self.device)

        z_mu, z_logvar = self.encode(X)
        z = self.reparameterize(z_mu, z_logvar)
        X_pred = self.decode(z)

        loss_recon = F.binary_cross_entropy(torch.sigmoid(X_pred), X)
        loss_reg = - 0.5 * torch.mean(
            torch.sum(
                1 + z_logvar - torch.exp(z_logvar) - z_mu ** 2,
                dim=list(range(1, z.ndim))
            )
        )

        total_loss = self.weight_recon * loss_recon + self.weight_reg * loss_reg

        return X_pred, total_loss, loss_recon, loss_reg
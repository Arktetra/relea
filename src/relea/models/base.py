from pathlib import Path
from typing import Any
from typing_extensions import override

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)

    def run_step(self, batch):
        raise NotImplementedError("Implement me!")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def save_checkpoint(self, path: Path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: Path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)
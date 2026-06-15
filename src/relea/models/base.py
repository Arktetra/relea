from pathlib import Path
from typing import Any
from typing_extensions import override
from safetensors.torch import save_file, load_file


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
        if(path.suffix == ".pt" or path.suffix == ".pth"): 
            torch.save(self.state_dict(), path)
        elif(path.suffix == ".safetensors"): 
            save_file(self.state_dict(), str(path))
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")

    def load_checkpoint(self, path: Path):
        if(path.suffix == ".pt" or path.suffix == ".pth"): 
            self.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        elif(path.suffix == ".safetensors"): 
            state_dict = load_file(str(path), device=self.device)
            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")
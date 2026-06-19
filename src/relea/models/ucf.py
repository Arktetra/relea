from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

from relea.models.base import BaseModule

import torch.nn as nn
import torch.nn.functional as F

class MViTVideoClassifier(BaseModule):
    def __init__(self, num_classes: int = 10, device: str = "cpu"):
        super().__init__()
        self.net = mvit_v2_s(weights=MViT_V2_S_Weights)
        in_features = self.net.head[1].in_features
        self.net.head[1] = nn.Linear(in_features=in_features, out_features=num_classes)

        self.loss = None
        self.device = device

    def forward(self, X):
        self.net(X)

    def run_step(self, batch):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        logits = self.net(X.permute(0, 2, 1, 3, 4))
        self.loss = F.cross_entropy(logits, y)
        return logits, self.loss
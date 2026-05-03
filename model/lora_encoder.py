import torch
import torch.nn as nn
import torch.nn.functional as F

class LoraEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, aggregator: nn.Module, normalize: bool = True):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)        # [B, T, D]
        z = self.aggregator(h)     # [B, D]
        return F.normalize(z, p=2, dim=-1) if self.normalize else z

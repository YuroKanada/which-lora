import torch
import torch.nn as nn


class MarginRankingTopKLoss(nn.Module):
    def __init__(self, margin: float = 0.1, topk: int = 1):
        super().__init__()
        self.margin = float(margin)
        self.topk = int(topk)
        if self.topk < 1:
            raise ValueError("topk must be >= 1")

    def _build_positive_mask(self, batch_size: int, device, labels=None) -> torch.Tensor:
        if labels is None:
            return torch.eye(batch_size, device=device, dtype=torch.bool)

        labels = [str(x) for x in labels]
        return torch.tensor(
            [[labels[i] == labels[j] for j in range(batch_size)] for i in range(batch_size)],
            device=device,
            dtype=torch.bool,
        )

    def forward(self, z_image: torch.Tensor, z_lora: torch.Tensor, labels=None) -> torch.Tensor:
        sims = z_image @ z_lora.t()
        batch_size = sims.size(0)
        positive_mask = self._build_positive_mask(batch_size, sims.device, labels=labels)

        positive_counts = positive_mask.sum(dim=1)
        valid_rows = positive_counts > 0
        if not torch.any(valid_rows):
            raise ValueError("No positive pairs found for ranking loss")

        s_pos = (sims * positive_mask.float()).sum(dim=1) / positive_counts.clamp_min(1).float()

        negative_sims = sims.masked_fill(positive_mask, float("-inf"))
        negative_counts = (~positive_mask).sum(dim=1)
        valid_rows = valid_rows & (negative_counts > 0)
        if not torch.any(valid_rows):
            raise ValueError("No negative pairs found for ranking loss")

        k = min(self.topk, batch_size - 1)
        topk_neg, _ = torch.topk(negative_sims, k=k, dim=1)
        s_neg = topk_neg.mean(dim=1)

        losses = torch.relu(self.margin - s_pos + s_neg)
        return losses[valid_rows].mean()

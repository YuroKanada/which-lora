# train/infonce.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE with optional multi-positive support.
    Inputs:
      z_text: [B, D]  (L2-normalized recommended)
      z_lora: [B, D]  (L2-normalized recommended)
    If labels is given, samples that share the same label are treated as positives.
    Otherwise positives are only (i, i).
    """
    def __init__(self, temperature: float = 0.07, symmetric: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)

    def _build_positive_mask(self, batch_size: int, device, labels=None) -> torch.Tensor:
        if labels is None:
            return torch.eye(batch_size, device=device, dtype=torch.bool)

        labels = [str(x) for x in labels]
        return torch.tensor(
            [[labels[i] == labels[j] for j in range(batch_size)] for i in range(batch_size)],
            device=device,
            dtype=torch.bool,
        )

    def _multi_positive_loss(self, logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        positive_counts = positive_mask.sum(dim=1)
        valid_rows = positive_counts > 0
        if not torch.any(valid_rows):
            raise ValueError("No positive pairs found for multi-positive InfoNCE")

        positive_log_probs = (log_probs * positive_mask.float()).sum(dim=1) / positive_counts.clamp_min(1).float()
        return -positive_log_probs[valid_rows].mean()

    def forward(self, z_text: torch.Tensor, z_lora: torch.Tensor, labels=None) -> torch.Tensor:
        logits = (z_text @ z_lora.t()) / self.temperature
        positive_mask = self._build_positive_mask(logits.size(0), logits.device, labels=labels)

        loss_t2l = self._multi_positive_loss(logits, positive_mask)
        if not self.symmetric:
            return loss_t2l

        loss_l2t = self._multi_positive_loss(logits.t(), positive_mask.t())
        return 0.5 * (loss_t2l + loss_l2t)

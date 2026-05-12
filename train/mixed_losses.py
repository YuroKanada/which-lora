import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMixGeometryLoss(nn.Module):
    """
    Encourage a mixed LoRA embedding to lie near the alpha-interpolation
    of its two parent LoRA embeddings.

    z_mixed:  [B, D]
    z_target: [B, D]
    z_donor:  [B, D]
    alpha:    [B] or [B, 1]

    alpha = 0   -> target side
    alpha = 1   -> donor side
    """

    def __init__(self, loss_type: str = "cosine"):
        super().__init__()
        if loss_type not in {"cosine", "mse"}:
            raise ValueError("loss_type must be 'cosine' or 'mse'")
        self.loss_type = loss_type

    def forward(
        self,
        z_mixed: torch.Tensor,
        z_target: torch.Tensor,
        z_donor: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)

        z_interp = (1.0 - alpha) * z_target + alpha * z_donor

        if self.loss_type == "mse":
            return F.mse_loss(z_mixed, z_interp)

        z_interp = F.normalize(z_interp, p=2, dim=-1)
        return (1.0 - F.cosine_similarity(z_mixed, z_interp, dim=-1)).mean()


class MonotonicMixRatioLoss(nn.Module):
    """
    Encourage embeddings with larger donor ratio alpha to be
    closer to donor and farther from target than embeddings with
    smaller donor ratio.

    This is a pairwise ranking-style loss over mixed LoRAs that share
    the same parent pair.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        z_mixed: torch.Tensor,
        z_target: torch.Tensor,
        z_donor: torch.Tensor,
        alpha: torch.Tensor,
        pair_ids,
    ) -> torch.Tensor:
        if z_mixed.size(0) == 0:
            return z_mixed.new_zeros(())

        alpha = alpha.view(-1)
        donor_sims = F.cosine_similarity(z_mixed, z_donor, dim=-1)
        target_sims = F.cosine_similarity(z_mixed, z_target, dim=-1)

        losses = []
        num_items = z_mixed.size(0)
        pair_ids = [str(x) for x in pair_ids]

        for i in range(num_items):
            for j in range(i + 1, num_items):
                if pair_ids[i] != pair_ids[j]:
                    continue
                if torch.isclose(alpha[i], alpha[j]):
                    continue

                if alpha[i] < alpha[j]:
                    low_idx, high_idx = i, j
                else:
                    low_idx, high_idx = j, i

                donor_term = F.relu(
                    self.margin
                    - donor_sims[high_idx]
                    + donor_sims[low_idx]
                )
                target_term = F.relu(
                    self.margin
                    - target_sims[low_idx]
                    + target_sims[high_idx]
                )
                losses.append(donor_term + target_term)

        if not losses:
            return z_mixed.new_zeros(())

        return torch.stack(losses).mean()


class MixedAwareRankingLoss(nn.Module):
    """
    Ranking loss for mixed retrieval where only related mixed LoRAs
    should be pushed above valid negatives.

    x_query:   [B, D] image/query embedding
    z_gallery: [N, D] gallery embeddings
    related_mixed_mask: [B, N] bool, True at related mixed LoRA positions
    ignore_mask: [B, N] bool, True at gallery positions that should be
                 excluded from ranking loss (e.g. source base LoRA)

    The loss pushes related mixed LoRAs above top-k valid negatives.
    """

    def __init__(self, margin: float = 0.1, topk: int = 1):
        super().__init__()
        self.margin = float(margin)
        self.topk = int(topk)
        if self.topk < 1:
            raise ValueError("topk must be >= 1")

    def forward(
        self,
        x_query: torch.Tensor,
        z_gallery: torch.Tensor,
        related_mixed_mask: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sims = x_query @ z_gallery.t()

        positive_mask = related_mixed_mask.bool()
        ignore_mask = (
            torch.zeros_like(positive_mask, dtype=torch.bool)
            if ignore_mask is None
            else ignore_mask.bool()
        )

        positive_counts = positive_mask.sum(dim=1)
        valid_rows = positive_counts > 0
        if not torch.any(valid_rows):
            raise ValueError("No related mixed positives found for mixed ranking loss")

        s_pos = (sims * positive_mask.float()).sum(dim=1) / positive_counts.clamp_min(1).float()

        excluded_mask = positive_mask | ignore_mask
        negative_sims = sims.masked_fill(excluded_mask, float("-inf"))
        negative_counts = (~excluded_mask).sum(dim=1)
        valid_rows = valid_rows & (negative_counts > 0)
        if not torch.any(valid_rows):
            raise ValueError("No valid negatives found for mixed ranking loss")

        k = min(self.topk, z_gallery.size(0) - 1)
        topk_neg, _ = torch.topk(negative_sims, k=k, dim=1)
        s_neg = topk_neg.mean(dim=1)

        losses = F.relu(self.margin - s_pos + s_neg)
        return losses[valid_rows].mean()

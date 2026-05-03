# model/heads.py
#学習対象としてtext_encoderの後のprojのみを設定する場合有
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, head_type: str = "linear",
                 hidden_dim: int = 1024, dropout: float = 0.0, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        if head_type == "linear":
            self.net = nn.Linear(in_dim, out_dim, bias=False)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim, bias=False),
            )
        else:
            raise ValueError(head_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1) if self.normalize else z

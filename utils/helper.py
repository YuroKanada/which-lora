import json
import os

import torch


def set_requires_grad(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def grad_norm(module):
    params = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return torch.norm(torch.stack([p.grad.detach().norm(2) for p in params]), 2).item()


def load_pretrained(module, path: str, strict: bool = True):
    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[WARN] load_state_dict({path})")
        if missing:
            print("  missing keys:", missing[:20], "..." if len(missing) > 20 else "")
        if unexpected:
            print("  unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")


def save_weights_best_only(
    out_dir: str,
    epoch: int,
    avg_loss: float,
    is_best: bool,
    image_enc,
    image_proj,
    lora_enc,
    lora_proj,
    save_lora_proj: bool = True,
    lora_encoder_mode: str = "full",
    lora_aggregator_type: str = "mlp",
    model_config_snapshot: dict | None = None,
):
    if not is_best:
        return

    os.makedirs(out_dir, exist_ok=True)

    torch.save(image_enc.state_dict(), os.path.join(out_dir, "image_enc.pt"))
    torch.save(image_proj.state_dict(), os.path.join(out_dir, "image_proj.pt"))
    torch.save(lora_enc.state_dict(), os.path.join(out_dir, "lora_enc.pt"))
    if save_lora_proj:
        torch.save(lora_proj.state_dict(), os.path.join(out_dir, "lora_proj.pt"))

    with open(os.path.join(out_dir, "best_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": int(epoch),
                "train_loss": float(avg_loss),
                "use_lora_proj": bool(save_lora_proj),
                "lora_encoder_mode": str(lora_encoder_mode),
                "lora_aggregator_type": str(lora_aggregator_type),
                "model_config": model_config_snapshot or {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] saved best weights -> {out_dir} (epoch={epoch}, loss={avg_loss:.6f})")

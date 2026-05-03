import json
import os

import torch

from config import CONFIG


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


def save_checkpoint(
    save_dir: str,
    epoch: int,
    avg_loss: float,
    is_best: bool,
    image_enc,
    image_proj,
    lora_enc,
    lora_proj,
    optimizer,
):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "train_loss": float(avg_loss),
        "config": CONFIG,
        "image_enc": image_enc.state_dict(),
        "image_proj": image_proj.state_dict(),
        "lora_enc": lora_enc.state_dict(),
        "lora_proj": lora_proj.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if not is_best:
        return

    best_path = os.path.join(save_dir, "best.pt")
    torch.save(checkpoint, best_path)

    meta_path = os.path.join(save_dir, "best_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"epoch": int(epoch), "train_loss": float(avg_loss)}, f, ensure_ascii=False, indent=2)


def save_weights_best_only(
    out_dir: str,
    epoch: int,
    avg_loss: float,
    is_best: bool,
    image_enc,
    image_proj,
    lora_enc,
    lora_proj,
):
    if not is_best:
        return

    os.makedirs(out_dir, exist_ok=True)

    torch.save(image_enc.state_dict(), os.path.join(out_dir, "image_enc.pt"))
    torch.save(image_proj.state_dict(), os.path.join(out_dir, "image_proj.pt"))
    torch.save(lora_enc.state_dict(), os.path.join(out_dir, "lora_enc.pt"))
    torch.save(lora_proj.state_dict(), os.path.join(out_dir, "lora_proj.pt"))

    with open(os.path.join(out_dir, "best_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": int(epoch), "train_loss": float(avg_loss)}, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved best weights -> {out_dir} (epoch={epoch}, loss={avg_loss:.6f})")

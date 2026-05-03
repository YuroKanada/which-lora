import os
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import wandb
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config import CONFIG
from dataset.collate import collate_image_lora
from dataset.image_lora_pairs import ImageLoRAPairsDataset
from dataset.loader import load_model_matrix
from model.aggregator import TokenAggregator
from model.heads import ProjectionHead
from model.image_encoder_factory import build_image_encoder
from model.lora_encoder import LoraEncoder
from model.transformer_encoder import TransformerEncoder
from train.infonce import InfoNCELoss
from utils.helper import (
    grad_norm,
    load_pretrained,
    save_checkpoint,
    save_weights_best_only,
    set_requires_grad,
)


DEVICE = CONFIG["device"]
timestamp = datetime.now().strftime("%Y%m%d%H%M")


def resolve_image_path_value(image_path_value, image_root_dir=None, path_replacements=None):
    image_path = Path(str(image_path_value)).expanduser()
    image_path_str = str(image_path)

    for src_prefix, dst_prefix in (path_replacements or []):
        src_prefix = str(src_prefix)
        dst_prefix = str(dst_prefix)
        if image_path_str.startswith(src_prefix):
            image_path = Path(image_path_str.replace(src_prefix, dst_prefix, 1)).expanduser()
            image_path_str = str(image_path)
            break

    if image_path.is_absolute():
        return str(image_path)
    if image_root_dir is not None:
        return str((Path(image_root_dir).expanduser().resolve() / image_path).resolve())
    return str(image_path)


def load_validation_artist_queries(jsonl_path, image_path_key, lora_id_key, image_root_dir=None, path_replacements=None):
    artist_to_positive_lora_ids = defaultdict(set)
    unique_queries = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if image_path_key not in item or lora_id_key not in item or "artist" not in item:
                raise ValueError(f"Invalid validation item: {item}")

            image_path = resolve_image_path_value(
                item[image_path_key],
                image_root_dir=image_root_dir,
                path_replacements=path_replacements,
            )
            artist = str(item["artist"])
            lora_id = str(item[lora_id_key])

            artist_to_positive_lora_ids[artist].add(lora_id)
            unique_queries[(image_path, artist)] = {"image_path": image_path, "artist": artist}

    return list(unique_queries.values()), {artist: sorted(ids) for artist, ids in artist_to_positive_lora_ids.items()}


@torch.no_grad()
def build_lora_gallery_embeddings(model_matrix_dict, lora_ids, lora_enc, lora_proj, device, batch_size):
    gallery_embeddings = []
    for start in range(0, len(lora_ids), batch_size):
        batch_ids = lora_ids[start : start + batch_size]
        batch_x = torch.stack([model_matrix_dict[lora_id] for lora_id in batch_ids], dim=0).to(device, non_blocking=True)
        batch_z = lora_proj(lora_enc(batch_x))
        gallery_embeddings.append(batch_z.detach().cpu())
    return torch.cat(gallery_embeddings, dim=0)


@torch.no_grad()
def evaluate_validation_retrieval(
    validation_queries,
    artist_to_positive_lora_ids,
    gallery_lora_ids,
    gallery_lora_embs,
    image_enc,
    image_proj,
    device,
    top_ks,
):
    ranks = []
    hits = {k: 0 for k in top_ks}
    num_rows = 0

    gallery_lora_embs = gallery_lora_embs.to(device, non_blocking=True)
    lora_id_to_index = {lora_id: idx for idx, lora_id in enumerate(gallery_lora_ids)}

    batch_size = CONFIG.get("validation_batch_size", CONFIG["batch_size"])
    for start in range(0, len(validation_queries), batch_size):
        batch_queries = validation_queries[start : start + batch_size]
        image_paths = [row["image_path"] for row in batch_queries]
        artists = [row["artist"] for row in batch_queries]

        valid_positions = []
        for batch_idx, artist in enumerate(artists):
            positive_lora_ids = [
                lora_id for lora_id in artist_to_positive_lora_ids.get(artist, [])
                if lora_id in lora_id_to_index
            ]
            if positive_lora_ids:
                valid_positions.append((batch_idx, artist, positive_lora_ids))
        if not valid_positions:
            continue

        batch_indices = [pos[0] for pos in valid_positions]
        batch_image_paths = [image_paths[idx] for idx in batch_indices]

        image_embs = image_proj(image_enc(batch_image_paths))
        sims = image_embs @ gallery_lora_embs.T
        order = torch.argsort(sims, dim=1, descending=True)

        for row_idx, (_, artist, positive_lora_ids) in enumerate(valid_positions):
            positive_indices = [lora_id_to_index[lora_id] for lora_id in positive_lora_ids]
            positive_ranks = [
                int((order[row_idx] == positive_idx).nonzero(as_tuple=False)[0].item()) + 1
                for positive_idx in positive_indices
            ]
            rank = min(positive_ranks)
            ranks.append(rank)
            num_rows += 1
            for k in top_ks:
                if rank <= k:
                    hits[k] += 1

    if num_rows == 0:
        return None

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    metrics = {
        "validation_num_queries": num_rows,
        "validation_artist_mean_best_rank": float(ranks_tensor.mean().item()),
        "validation_artist_median_best_rank": float(ranks_tensor.median().item()),
    }
    for k in top_ks:
        metrics[f"validation_artist_hit@{k}"] = hits[k] / num_rows
    return metrics

if CONFIG.get("wandb_api_key"):
    os.environ["WANDB_API_KEY"] = CONFIG["wandb_api_key"]

wandb.init(
    project=CONFIG.get("wandb_project", "image_lora_retrieval"),
    name=f"{CONFIG.get('wandb_run_name_prefix', 'run')}_{timestamp}",
    config=CONFIG,
)
print("[DEBUG] start main, epochs =", CONFIG["epochs"])

model_matrix_dict = load_model_matrix(CONFIG["category_dir"], CONFIG["dimension"])

train_ds = ImageLoRAPairsDataset(
    CONFIG["train_pairs_jsonl"],
    image_path_key=CONFIG.get("image_path_key", "image_path"),
    lora_id_key=CONFIG.get("lora_id_key", "lora_id"),
    image_root_dir=CONFIG.get("image_root_dir"),
    path_replacements=CONFIG.get("image_path_replacements"),
)
train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG.get("num_workers", 4),
    pin_memory=True,
    collate_fn=lambda batch: collate_image_lora(batch, model_matrix_dict),
)

validation_queries = None
artist_to_positive_lora_ids = None
validation_pairs_jsonl = CONFIG.get("validation_pairs_jsonl")
if validation_pairs_jsonl and Path(validation_pairs_jsonl).exists():
    validation_queries, artist_to_positive_lora_ids = load_validation_artist_queries(
        validation_pairs_jsonl,
        image_path_key=CONFIG.get("image_path_key", "image_path"),
        lora_id_key=CONFIG.get("lora_id_key", "lora_id"),
        image_root_dir=CONFIG.get("image_root_dir"),
        path_replacements=CONFIG.get("image_path_replacements"),
    )
    print(f"[DEBUG] validation unique queries = {len(validation_queries)}")
    print(f"[DEBUG] validation artists = {len(artist_to_positive_lora_ids)}")

image_enc = build_image_encoder(CONFIG, DEVICE).to(DEVICE)

if CONFIG.get("freeze_image", True):
    image_enc.freeze_image(CONFIG.get("train_image_projection", False))
else:
    image_enc.unfreeze_image(CONFIG.get("train_image_projection", False))

image_proj = ProjectionHead(
    in_dim=image_enc.clip_dim,
    out_dim=CONFIG["dimension"],
    head_type=CONFIG["image_proj_type"],
    hidden_dim=CONFIG["image_proj_hidden_dim"],
    dropout=CONFIG["image_proj_dropout"],
    normalize=CONFIG.get("image_proj_normalize", True),
).to(DEVICE)
set_requires_grad(image_proj, not CONFIG.get("freeze_image_proj", False))

encoder = TransformerEncoder(
    d_model=CONFIG["dimension"],
    num_layers=CONFIG["num_layers"],
    num_heads=CONFIG["num_heads"],
    d_ff=CONFIG["ff_dim"],
    max_len=CONFIG["max_len"],
    use_pos_emb=CONFIG["use_positional_embedding"],
)
aggregator = TokenAggregator(
    CONFIG["dimension"],
    hidden_dim=CONFIG["aggregator_hidden_dim"],
    use_mlp=CONFIG["use_mlp_aggregator"],
)

if CONFIG.get("pretrained_lora_encoder", False):
    load_pretrained(encoder, CONFIG["pretrained_encoder_path"], strict=CONFIG.get("pretrained_strict", True))
    load_pretrained(aggregator, CONFIG["pretrained_aggregator_path"], strict=CONFIG.get("pretrained_strict", True))
    print("[OK] loaded pretrained LoRA encoder weights")

encoder = encoder.to(DEVICE)
aggregator = aggregator.to(DEVICE)

lora_enc = LoraEncoder(encoder, aggregator, normalize=True).to(DEVICE)
set_requires_grad(lora_enc, not CONFIG.get("freeze_lora_encoder", False))

lora_proj = ProjectionHead(
    in_dim=CONFIG["dimension"],
    out_dim=CONFIG["dimension"],
    head_type=CONFIG.get("lora_proj_type", "linear"),
    hidden_dim=CONFIG.get("lora_proj_hidden_dim", 512),
    dropout=CONFIG.get("lora_proj_dropout", 0.0),
    normalize=True,
).to(DEVICE)
set_requires_grad(lora_proj, not CONFIG.get("freeze_lora_proj", False))

loss_fn = InfoNCELoss(
    temperature=CONFIG.get("infonce_temperature", 0.07),
    symmetric=True,
).to(DEVICE)

param_groups = [
    {"params": [p for p in image_enc.parameters() if p.requires_grad], "lr": CONFIG["lr"], "name": "image_enc"},
    {"params": [p for p in image_proj.parameters() if p.requires_grad], "lr": CONFIG["lr"], "name": "image_proj"},
    {"params": [p for p in lora_enc.parameters() if p.requires_grad], "lr": CONFIG["lr"], "name": "lora_enc"},
    {"params": [p for p in lora_proj.parameters() if p.requires_grad], "lr": CONFIG["lr"], "name": "lora_proj"},
]
param_groups = [group for group in param_groups if group["params"]]
optimizer = torch.optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])

wandb.config.update(
    {
        "optimizer": "AdamW",
        "lr": CONFIG["lr"],
        "weight_decay": CONFIG["weight_decay"],
        "infonce_temperature": CONFIG.get("infonce_temperature", 0.07),
        "freeze_image": CONFIG.get("freeze_image", True),
        "freeze_image_proj": CONFIG.get("freeze_image_proj", False),
        "freeze_lora_encoder": CONFIG.get("freeze_lora_encoder", False),
        "freeze_lora_proj": CONFIG.get("freeze_lora_proj", True),
        "pretrained_lora_encoder": CONFIG.get("pretrained_lora_encoder", False),
        "image_encoder_type": CONFIG.get("image_encoder_type", "openclip"),
        "image_encoder_model_name": CONFIG.get("image_encoder_model_name"),
        "image_encoder_pretrained": CONFIG.get("image_encoder_pretrained"),
        "validation_pairs_jsonl": validation_pairs_jsonl,
    }
)

CKPT_DIR = os.path.join(CONFIG["checkpoint_dir"], timestamp)  # 復旧用（bestのみ）
SAVED_DIR = os.path.join(CONFIG["save_dir"], timestamp)
best_train_loss = float("inf")

for epoch in range(CONFIG["epochs"]):
    print("[DEBUG] enter epoch", epoch + 1)

    image_enc.train(not CONFIG.get("freeze_image", True))
    image_proj.train(not CONFIG.get("freeze_image_proj", False))
    lora_enc.train(not CONFIG.get("freeze_lora_encoder", False))
    lora_proj.train(not CONFIG.get("freeze_lora_proj", False))

    total_loss = 0.0
    n = 0

    for image_paths, lora_x, lora_ids in train_loader:
        lora_x = lora_x.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        zi_raw = image_enc(image_paths)
        zi = image_proj(zi_raw)

        zl = lora_enc(lora_x)
        zl = lora_proj(zl)

        loss = loss_fn(zi, zl, labels=lora_ids)
        loss.backward()

        if CONFIG.get("grad_clip_norm", 0.0) > 0:
            clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                CONFIG["grad_clip_norm"],
            )

        optimizer.step()

        batch_size = lora_x.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size

    avg_loss = total_loss / max(n, 1)

    log_dict = {"epoch": epoch + 1, "train_loss": avg_loss}
    for i, group in enumerate(optimizer.param_groups):
        log_dict[f"lr/group_{i}"] = group["lr"]

    log_dict["grad_norm/image_proj"] = grad_norm(image_proj)
    log_dict["grad_norm/lora_enc"] = grad_norm(lora_enc)
    log_dict["grad_norm/lora_proj"] = grad_norm(lora_proj)

    run_validation = (
        validation_queries is not None
        and CONFIG.get("validation_eval_every", 1) > 0
        and (epoch + 1) % CONFIG.get("validation_eval_every", 1) == 0
    )
    validation_metrics = None
    if run_validation:
        image_enc.eval()
        image_proj.eval()
        lora_enc.eval()
        lora_proj.eval()

        gallery_lora_ids = sorted(model_matrix_dict.keys())
        gallery_lora_embs = build_lora_gallery_embeddings(
            model_matrix_dict=model_matrix_dict,
            lora_ids=gallery_lora_ids,
            lora_enc=lora_enc,
            lora_proj=lora_proj,
            device=DEVICE,
            batch_size=CONFIG.get("validation_batch_size", CONFIG["batch_size"]),
        )
        validation_metrics = evaluate_validation_retrieval(
            validation_queries=validation_queries,
            artist_to_positive_lora_ids=artist_to_positive_lora_ids,
            gallery_lora_ids=gallery_lora_ids,
            gallery_lora_embs=gallery_lora_embs,
            image_enc=image_enc,
            image_proj=image_proj,
            device=DEVICE,
            top_ks=tuple(CONFIG.get("validation_top_ks", [1, 5, 10])),
        )
        if validation_metrics is not None:
            log_dict.update(validation_metrics)

    wandb.log(log_dict)

    is_best = avg_loss < best_train_loss
    if is_best:
        best_train_loss = avg_loss

    save_checkpoint(
        save_dir=CKPT_DIR,
        epoch=epoch + 1,
        avg_loss=avg_loss,
        is_best=is_best,
        image_enc=image_enc,
        image_proj=image_proj,
        lora_enc=lora_enc,
        lora_proj=lora_proj,
        optimizer=optimizer,
    )

    save_weights_best_only(
        out_dir=SAVED_DIR,
        epoch=epoch + 1,
        avg_loss=avg_loss,
        is_best=is_best,
        image_enc=image_enc,
        image_proj=image_proj,
        lora_enc=lora_enc,
        lora_proj=lora_proj,
    )

    message = f"[epoch {epoch + 1:02d}] train_loss={avg_loss:.6f} best={best_train_loss:.6f}"
    if validation_metrics is not None:
        metric_parts = [
            f"val_artist_hit@{k}={validation_metrics[f'validation_artist_hit@{k}']:.4f}"
            for k in CONFIG.get("validation_top_ks", [1, 5, 10])
            if f"validation_artist_hit@{k}" in validation_metrics
        ]
        metric_parts.append(f"val_artist_median_rank={validation_metrics['validation_artist_median_best_rank']:.2f}")
        message += " " + " ".join(metric_parts)
    print(message)

wandb.finish()

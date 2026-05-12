import os
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from itertools import cycle

import wandb
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config import CONFIG
from dataset.collate import collate_image_lora, collate_image_mixed_relations
from dataset.image_lora_pairs import ImageLoRAPairsDataset
from dataset.image_mixed_relations import ImageMixedRelationsDataset
from dataset.loader import load_model_matrix
from model.aggregator import TokenAggregator
from model.heads import ProjectionHead
from model.image_encoder_factory import build_image_encoder
from model.lora_encoder import LoraEncoder
from model.transformer_encoder import TransformerEncoder
from train.infonce import InfoNCELoss
from train.ranking import MarginRankingTopKLoss
from train.mixed_losses import MixedAwareRankingLoss
from utils.helper import (
    grad_norm,
    load_pretrained,
    save_weights_best_only,
    set_requires_grad,
)


DEVICE = CONFIG["device"]
timestamp = datetime.now().strftime("%Y%m%d%H%M")
IMAGE_PATH_KEY = "image_path"
LORA_ID_KEY = "lora_id"
VALIDATION_BATCH_SIZE = CONFIG.get("validation_batch_size", CONFIG["batch_size"])
EVAL100_TOP_KS = (1, 5, 10)
MIXED_EVAL_TOP_KS = (5, 10, 20)


def canonicalize_lora_id(lora_id: str) -> str:
    text = str(lora_id)
    if ":" in text:
        text = text.split(":", 1)[1]
    if text.isdigit():
        text = f"lora_{text.zfill(6)}"
    if text.startswith("lora_"):
        return text
    return text


def build_gallery_entries(base_lora_ids, mixed_lora_ids):
    entries = []
    for lora_id in base_lora_ids:
        entries.append(("base", canonicalize_lora_id(lora_id)))
    for lora_id in mixed_lora_ids:
        entries.append(("mixed", canonicalize_lora_id(lora_id)))
    return entries


def build_typed_lora_id_mask(lora_id_rows, gallery_entries, gallery_type):
    gallery_index = {
        (entry_type, entry_id): idx
        for idx, (entry_type, entry_id) in enumerate(gallery_entries)
    }
    mask = torch.zeros((len(lora_id_rows), len(gallery_entries)), dtype=torch.bool)

    for row_idx, row_ids in enumerate(lora_id_rows):
        for lora_id in row_ids:
            key = (gallery_type, canonicalize_lora_id(lora_id))
            if key in gallery_index:
                mask[row_idx, gallery_index[key]] = True

    return mask


def load_base_artist_groups(plan_csv_path):
    base_lora_to_artist = {}
    artist_to_base_lora_ids = defaultdict(set)

    with open(plan_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lora_id = canonicalize_lora_id(str(row["lora_id"]))
            artist = str(row["artist"])
            base_lora_to_artist[lora_id] = artist
            artist_to_base_lora_ids[artist].add(lora_id)

    artist_to_base_lora_ids = {
        artist: sorted(lora_ids)
        for artist, lora_ids in artist_to_base_lora_ids.items()
    }
    return base_lora_to_artist, artist_to_base_lora_ids


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


def load_eval_pair_queries(jsonl_path, image_path_key, lora_id_key, image_root_dir=None, path_replacements=None):
    queries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if image_path_key not in item or lora_id_key not in item:
                raise ValueError(f"Invalid eval pair item: {item}")
            queries.append(
                {
                    "image_path": resolve_image_path_value(
                        item[image_path_key],
                        image_root_dir=image_root_dir,
                        path_replacements=path_replacements,
                    ),
                    "lora_id": str(item[lora_id_key]),
                }
            )
    return queries


def load_mixed_eval_queries(jsonl_path, mixed_plan_csv, mixed_eval_ids_json):
    with open(mixed_eval_ids_json, "r", encoding="utf-8") as f:
        mixed_eval_ids = set(json.load(f))

    donor_artist_to_ratio_ids = defaultdict(lambda: defaultdict(set))
    with open(mixed_plan_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mixed_id = f"mixed:{row['lora_id']}"
            if mixed_id not in mixed_eval_ids:
                continue
            donor_artist = str(row["donor_artist"])
            ratio_tag = str(row["mix_ratio_tag"])
            donor_artist_to_ratio_ids[donor_artist][ratio_tag].add(mixed_id)

    queries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            donor_artist = str(item["artist"])
            ratio_map = {
                ratio: sorted(ids)
                for ratio, ids in donor_artist_to_ratio_ids.get(donor_artist, {}).items()
                if ids
            }
            all_positive_ids = sorted({mid for ids in ratio_map.values() for mid in ids})
            if not all_positive_ids:
                continue
            queries.append(
                {
                    "image_path": str(item["source_image_path"]),
                    "artist": donor_artist,
                    "positive_lora_ids": all_positive_ids,
                    "ratio_to_positive_ids": ratio_map,
                }
            )
    return queries


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
def evaluate_pair_retrieval(
    validation_queries,
    gallery_lora_ids,
    gallery_lora_embs,
    image_enc,
    image_proj,
    device,
    top_ks,
    prefix,
):
    if not validation_queries:
        return None

    gallery_lora_embs = gallery_lora_embs.to(device, non_blocking=True)
    lora_id_to_index = {lora_id: idx for idx, lora_id in enumerate(gallery_lora_ids)}

    ranks = []
    hits = {k: 0 for k in top_ks}
    batch_size = VALIDATION_BATCH_SIZE
    for start in range(0, len(validation_queries), batch_size):
        batch_queries = validation_queries[start : start + batch_size]
        valid_queries = [q for q in batch_queries if q["lora_id"] in lora_id_to_index]
        if not valid_queries:
            continue
        image_paths = [q["image_path"] for q in valid_queries]
        positive_indices = [lora_id_to_index[q["lora_id"]] for q in valid_queries]
        image_embs = image_proj(image_enc(image_paths))
        sims = image_embs @ gallery_lora_embs.T
        order = torch.argsort(sims, dim=1, descending=True)
        for row_idx, positive_idx in enumerate(positive_indices):
            rank = int((order[row_idx] == positive_idx).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
            for k in top_ks:
                if rank <= k:
                    hits[k] += 1

    if not ranks:
        return None

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    reciprocal = (1.0 / ranks_tensor).mean().item()
    metrics = {
        f"{prefix}_num_queries": len(ranks),
        f"{prefix}_mean_rank": float(ranks_tensor.mean().item()),
        f"{prefix}_median_rank": float(ranks_tensor.median().item()),
        f"{prefix}_mrr": float(reciprocal),
    }
    for k in top_ks:
        metrics[f"{prefix}_hit@{k}"] = hits[k] / len(ranks)
    return metrics


@torch.no_grad()
def evaluate_multi_positive_retrieval(
    queries,
    gallery_lora_ids,
    gallery_lora_embs,
    image_enc,
    image_proj,
    device,
    top_ks,
    prefix,
):
    if not queries:
        return None

    gallery_lora_embs = gallery_lora_embs.to(device, non_blocking=True)
    lora_id_to_index = {lora_id: idx for idx, lora_id in enumerate(gallery_lora_ids)}
    batch_size = VALIDATION_BATCH_SIZE

    hits = {k: 0 for k in top_ks}
    recalls = {k: 0.0 for k in top_ks}
    ratio_hits = defaultdict(lambda: {k: 0 for k in top_ks})
    ratio_recalls = defaultdict(lambda: {k: 0.0 for k in top_ks})
    ratio_counts = defaultdict(int)
    num_queries = 0

    for start in range(0, len(queries), batch_size):
        batch_queries = queries[start : start + batch_size]
        valid_rows = []
        for query in batch_queries:
            positive_indices = [
                lora_id_to_index[lora_id]
                for lora_id in query["positive_lora_ids"]
                if lora_id in lora_id_to_index
            ]
            ratio_to_indices = {}
            for ratio, lora_ids in query["ratio_to_positive_ids"].items():
                indices = [lora_id_to_index[lora_id] for lora_id in lora_ids if lora_id in lora_id_to_index]
                if indices:
                    ratio_to_indices[ratio] = indices
            if positive_indices:
                valid_rows.append((query["image_path"], positive_indices, ratio_to_indices))
        if not valid_rows:
            continue

        image_paths = [row[0] for row in valid_rows]
        image_embs = image_proj(image_enc(image_paths))
        sims = image_embs @ gallery_lora_embs.T
        order = torch.argsort(sims, dim=1, descending=True)

        for row_idx, (_, positive_indices, ratio_to_indices) in enumerate(valid_rows):
            order_row = order[row_idx]
            num_queries += 1
            positive_set = set(positive_indices)
            for k in top_ks:
                topk_indices = set(order_row[:k].tolist())
                hit = int(bool(positive_set & topk_indices))
                hits[k] += hit
                recalls[k] += len(positive_set & topk_indices) / len(positive_set)

            for ratio, indices in ratio_to_indices.items():
                ratio_counts[ratio] += 1
                index_set = set(indices)
                for k in top_ks:
                    topk_indices = set(order_row[:k].tolist())
                    hit = int(bool(index_set & topk_indices))
                    ratio_hits[ratio][k] += hit
                    ratio_recalls[ratio][k] += len(index_set & topk_indices) / len(index_set)

    if num_queries == 0:
        return None

    metrics = {f"{prefix}_num_queries": num_queries}
    for k in top_ks:
        metrics[f"{prefix}_hit@{k}"] = hits[k] / num_queries
        metrics[f"{prefix}_recall@{k}"] = recalls[k] / num_queries
    for ratio, count in ratio_counts.items():
        if count == 0:
            continue
        for k in top_ks:
            metrics[f"{prefix}_{ratio}_hit@{k}"] = ratio_hits[ratio][k] / count
            metrics[f"{prefix}_{ratio}_recall@{k}"] = ratio_recalls[ratio][k] / count
    return metrics


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

    batch_size = VALIDATION_BATCH_SIZE
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

train_model_matrix_dict = load_model_matrix(
    CONFIG["category_dir"],
    CONFIG["dimension"],
    namespace=CONFIG.get("base_lora_namespace"),
    add_lora_prefix=bool(CONFIG.get("base_lora_namespace")),
)
model_matrix_dict = train_model_matrix_dict
base_model_matrix_dict = {
    lora_id: mat for lora_id, mat in train_model_matrix_dict.items()
    if not str(lora_id).startswith("mixed:")
}
mixed_model_matrix_dict = {
    lora_id: mat for lora_id, mat in train_model_matrix_dict.items()
    if str(lora_id).startswith("mixed:")
}
if CONFIG.get("use_mixed_rank_loss", False) and not mixed_model_matrix_dict:
    mixed_model_matrix_dict = load_model_matrix(
        CONFIG["category_dir"],
        CONFIG["dimension"],
        namespace=CONFIG.get("mixed_lora_namespace"),
        add_lora_prefix=bool(CONFIG.get("mixed_lora_namespace")),
    )
    base_model_matrix_dict = model_matrix_dict

image_enc = build_image_encoder(CONFIG, DEVICE).to(DEVICE)

train_ds = ImageLoRAPairsDataset(
    CONFIG["train_pairs_jsonl"],
    image_path_key=IMAGE_PATH_KEY,
    lora_id_key=LORA_ID_KEY,
    image_root_dir=CONFIG.get("image_root_dir"),
    path_replacements=CONFIG.get("image_path_replacements"),
)
train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG.get("num_workers", 4),
    pin_memory=True,
    collate_fn=lambda batch: collate_image_lora(batch, train_model_matrix_dict, image_preprocess=image_enc.preprocess),
)

mixed_train_loader = None
mixed_gallery_entries = None
mixed_gallery_x = None
base_lora_to_artist = {}
artist_to_base_lora_ids = {}
if CONFIG.get("use_mixed_rank_loss", False):
    base_lora_to_artist, artist_to_base_lora_ids = load_base_artist_groups(
        CONFIG["base_plan_csv"]
    )
    mixed_train_ds = ImageMixedRelationsDataset(
        CONFIG["mixed_train_relations_jsonl"],
        image_path_key=IMAGE_PATH_KEY,
        base_lora_id_key="base_lora_id",
        source_type_key="source_type",
        related_mixed_ids_key="related_mixed_lora_ids",
        base_related_mixed_ids_key="base_related_mixed_lora_ids",
        image_root_dirs=CONFIG.get("mixed_image_root_dirs"),
        path_replacements=CONFIG.get("image_path_replacements"),
    )
    mixed_train_loader = DataLoader(
        mixed_train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG.get("num_workers", 4),
        pin_memory=True,
        collate_fn=lambda batch: collate_image_mixed_relations(batch, image_preprocess=image_enc.preprocess),
    )
    base_gallery_lora_ids = sorted(base_model_matrix_dict.keys())
    mixed_gallery_lora_ids = sorted(mixed_model_matrix_dict.keys())
    mixed_gallery_entries = build_gallery_entries(base_gallery_lora_ids, mixed_gallery_lora_ids)
    mixed_gallery_x = torch.stack(
        [base_model_matrix_dict[lora_id] for lora_id in base_gallery_lora_ids]
        + [mixed_model_matrix_dict[lora_id] for lora_id in mixed_gallery_lora_ids],
        dim=0,
    )

eval100_queries = None
eval100_model_matrix_dict = None
eval100_pairs_jsonl = CONFIG.get("eval100_pairs_jsonl")
if eval100_pairs_jsonl and Path(eval100_pairs_jsonl).exists():
    eval100_queries = load_eval_pair_queries(
        eval100_pairs_jsonl,
        image_path_key=IMAGE_PATH_KEY,
        lora_id_key=LORA_ID_KEY,
        image_root_dir=CONFIG.get("image_root_dir"),
        path_replacements=CONFIG.get("image_path_replacements"),
    )
    eval100_model_matrix_dict = load_model_matrix(
        CONFIG["eval100_category_dir"],
        CONFIG["dimension"],
        namespace="base",
        add_lora_prefix=True,
    )
    print(f"[DEBUG] eval100 queries = {len(eval100_queries)}")
    print(f"[DEBUG] eval100 gallery = {len(eval100_model_matrix_dict)}")

mixed_eval_queries = None
mixed_eval_gallery_entries = None
mixed_eval_gallery_x = None
if (
    CONFIG.get("mixed_eval_queries_jsonl")
    and CONFIG.get("mixed_eval_ids_json")
    and Path(CONFIG["mixed_eval_queries_jsonl"]).exists()
    and Path(CONFIG["mixed_eval_ids_json"]).exists()
):
    mixed_eval_queries = load_mixed_eval_queries(
        CONFIG["mixed_eval_queries_jsonl"],
        CONFIG["mixed_eval_plan_csv"],
        CONFIG["mixed_eval_ids_json"],
    )
    mixed_eval_model_matrix_dict = load_model_matrix(
        CONFIG["mixed_eval_category_dir"],
        CONFIG["dimension"],
        namespace="mixed",
        add_lora_prefix=True,
    )
    if eval100_model_matrix_dict is None:
        eval100_model_matrix_dict = load_model_matrix(
            CONFIG["eval100_category_dir"],
            CONFIG["dimension"],
            namespace="base",
            add_lora_prefix=True,
        )
    mixed_eval_base_ids = sorted(eval100_model_matrix_dict.keys())
    mixed_eval_mixed_ids = sorted(mixed_eval_model_matrix_dict.keys())
    mixed_eval_gallery_entries = build_gallery_entries(mixed_eval_base_ids, mixed_eval_mixed_ids)
    mixed_eval_gallery_x = torch.stack(
        [eval100_model_matrix_dict[lora_id] for lora_id in mixed_eval_base_ids]
        + [mixed_eval_model_matrix_dict[lora_id] for lora_id in mixed_eval_mixed_ids],
        dim=0,
    )
    print(f"[DEBUG] mixed eval queries = {len(mixed_eval_queries)}")
    print(f"[DEBUG] mixed eval gallery = {len(mixed_eval_gallery_entries)}")

if CONFIG.get("freeze_image", True):
    image_enc.freeze_image(False)
else:
    image_enc.unfreeze_image(False)

image_proj = ProjectionHead(
    in_dim=image_enc.clip_dim,
    out_dim=CONFIG["dimension"],
    head_type=CONFIG["image_proj_type"],
    hidden_dim=CONFIG["image_proj_hidden_dim"],
    dropout=CONFIG["image_proj_dropout"],
    normalize=True,
).to(DEVICE)
set_requires_grad(image_proj, not CONFIG.get("freeze_image_proj", False))

encoder = TransformerEncoder(
    d_model=CONFIG["dimension"],
    num_layers=CONFIG["num_layers"],
    num_heads=CONFIG["num_heads"],
    d_ff=CONFIG["ff_dim"],
    max_len=264,
    use_pos_emb=True,
)
aggregator = TokenAggregator(
    CONFIG["dimension"],
    hidden_dim=64,
    use_mlp=True,
)

if CONFIG.get("pretrained_lora_encoder", False):
    load_pretrained(encoder, CONFIG["pretrained_encoder_path"], strict=True)
    load_pretrained(aggregator, CONFIG["pretrained_aggregator_path"], strict=True)
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
rank_loss_fn = MarginRankingTopKLoss(
    margin=CONFIG.get("rank_loss_margin", 0.1),
    topk=CONFIG.get("rank_topk", 1),
).to(DEVICE)
mixed_rank_loss_fn = MixedAwareRankingLoss(
    margin=CONFIG.get("mixed_rank_loss_margin", 0.1),
    topk=CONFIG.get("mixed_rank_topk", 1),
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
        "train_pairs_jsonl": CONFIG["train_pairs_jsonl"],
        "category_dir": CONFIG["category_dir"],
        "eval100_pairs_jsonl": eval100_pairs_jsonl,
        "eval100_category_dir": CONFIG["eval100_category_dir"],
        "mixed_eval_queries_jsonl": CONFIG.get("mixed_eval_queries_jsonl"),
        "mixed_eval_category_dir": CONFIG["mixed_eval_category_dir"],
    }
)

SAVED_DIR = os.path.join(CONFIG["save_dir"], timestamp)
best_train_loss = float("inf")

for epoch in range(CONFIG["epochs"]):
    print("[DEBUG] enter epoch", epoch + 1)
    epoch_start_time = time.perf_counter()
    mixed_rank_active = (
        CONFIG.get("use_mixed_rank_loss", False)
        and epoch >= int(CONFIG.get("mixed_rank_warmup_epochs", 0))
    )

    image_enc.train(not CONFIG.get("freeze_image", True))
    image_proj.train(not CONFIG.get("freeze_image_proj", False))
    lora_enc.train(not CONFIG.get("freeze_lora_encoder", False))
    lora_proj.train(not CONFIG.get("freeze_lora_proj", False))

    total_loss = 0.0
    total_infonce_loss = 0.0
    total_rank_loss = 0.0
    total_mixed_rank_loss = 0.0
    n = 0
    mixed_n = 0
    mixed_train_iter = cycle(mixed_train_loader) if mixed_train_loader is not None else None

    for image_inputs, lora_x, lora_ids in train_loader:
        lora_x = lora_x.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        zi_raw = image_enc(image_inputs)
        zi = image_proj(zi_raw)

        zl = lora_enc(lora_x)
        zl = lora_proj(zl)

        infonce_loss = loss_fn(zi, zl, labels=lora_ids)
        rank_loss = torch.zeros((), device=DEVICE)
        mixed_rank_loss = torch.zeros((), device=DEVICE)
        mixed_batch_size = 0
        loss = infonce_loss
        #Trueでranking lossもロスに追加
        if CONFIG.get("use_rank_loss", False):
            rank_loss = rank_loss_fn(zi, zl, labels=lora_ids)
            loss = loss + CONFIG.get("rank_loss_weight", 0.1) * rank_loss

        if mixed_rank_active and mixed_train_iter is not None:
            mixed_image_inputs, mixed_base_lora_ids, _, related_mixed_lora_ids, base_related_mixed_lora_ids = next(mixed_train_iter)
            mixed_query_embs = image_proj(image_enc(mixed_image_inputs))
            gallery_x = mixed_gallery_x.to(DEVICE, non_blocking=True)
            mixed_gallery_embs = lora_proj(lora_enc(gallery_x))
            related_mixed_mask = build_typed_lora_id_mask(
                related_mixed_lora_ids,
                mixed_gallery_entries,
                gallery_type="mixed",
            ).to(DEVICE, non_blocking=True)
            base_related_mixed_mask = build_typed_lora_id_mask(
                base_related_mixed_lora_ids,
                mixed_gallery_entries,
                gallery_type="mixed",
            ).to(DEVICE, non_blocking=True)
            source_base_mask = build_typed_lora_id_mask(
                [[base_lora_id] for base_lora_id in mixed_base_lora_ids],
                mixed_gallery_entries,
                gallery_type="base",
            ).to(DEVICE, non_blocking=True)
            same_artist_base_ids = []
            for base_lora_id in mixed_base_lora_ids:
                canonical_base_lora_id = canonicalize_lora_id(base_lora_id)
                artist = base_lora_to_artist.get(canonical_base_lora_id)
                if artist is None:
                    same_artist_base_ids.append([base_lora_id])
                    continue
                same_artist_base_ids.append(artist_to_base_lora_ids.get(artist, [canonical_base_lora_id]))
            same_artist_base_mask = build_typed_lora_id_mask(
                same_artist_base_ids,
                mixed_gallery_entries,
                gallery_type="base",
            ).to(DEVICE, non_blocking=True)
            mixed_rank_loss = mixed_rank_loss_fn(
                mixed_query_embs,
                mixed_gallery_embs,
                related_mixed_mask=related_mixed_mask,
                ignore_mask=(base_related_mixed_mask | source_base_mask | same_artist_base_mask),
            )
            loss = loss + CONFIG.get("mixed_rank_loss_weight", 0.1) * mixed_rank_loss
            mixed_batch_size = mixed_query_embs.size(0)
            mixed_n += mixed_batch_size

        loss.backward()

        if CONFIG.get("grad_clip_norm", 0.0) > 0:
            clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                CONFIG["grad_clip_norm"],
            )

        optimizer.step()

        batch_size = lora_x.size(0)
        total_loss += loss.item() * batch_size
        total_infonce_loss += infonce_loss.item() * batch_size
        total_rank_loss += rank_loss.item() * batch_size
        total_mixed_rank_loss += mixed_rank_loss.item() * (
            mixed_batch_size if mixed_batch_size > 0 else batch_size
        )
        n += batch_size

    avg_loss = total_loss / max(n, 1)
    avg_infonce_loss = total_infonce_loss / max(n, 1)
    avg_rank_loss = total_rank_loss / max(n, 1)
    avg_mixed_rank_loss = total_mixed_rank_loss / max(mixed_n, 1)
    epoch_time_sec = time.perf_counter() - epoch_start_time
    train_samples_per_sec = n / max(epoch_time_sec, 1e-8)

    log_dict = {
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_infonce_loss": avg_infonce_loss,
        "train_rank_loss": avg_rank_loss,
        "train_mixed_rank_loss": avg_mixed_rank_loss,
        "mixed_rank_active": int(mixed_rank_active),
        "epoch_time_sec": epoch_time_sec,
        "train_samples_per_sec": train_samples_per_sec,
    }
    for i, group in enumerate(optimizer.param_groups):
        log_dict[f"lr/group_{i}"] = group["lr"]

    log_dict["grad_norm/image_proj"] = grad_norm(image_proj)
    log_dict["grad_norm/lora_enc"] = grad_norm(lora_enc)
    log_dict["grad_norm/lora_proj"] = grad_norm(lora_proj)

    run_eval = (
        CONFIG.get("validation_eval_every", 1) > 0
        and (epoch + 1) % CONFIG.get("validation_eval_every", 1) == 0
    )
    eval100_metrics = None
    mixed_eval_metrics = None
    if run_eval and (eval100_queries is not None or mixed_eval_queries is not None):
        image_enc.eval()
        image_proj.eval()
        lora_enc.eval()
        lora_proj.eval()

        if eval100_queries is not None and eval100_model_matrix_dict is not None:
            eval100_lora_ids = sorted(eval100_model_matrix_dict.keys())
            eval100_lora_embs = build_lora_gallery_embeddings(
                model_matrix_dict=eval100_model_matrix_dict,
                lora_ids=eval100_lora_ids,
                lora_enc=lora_enc,
                lora_proj=lora_proj,
                device=DEVICE,
                batch_size=VALIDATION_BATCH_SIZE,
            )
            eval100_metrics = evaluate_pair_retrieval(
                validation_queries=eval100_queries,
                gallery_lora_ids=eval100_lora_ids,
                gallery_lora_embs=eval100_lora_embs,
                image_enc=image_enc,
                image_proj=image_proj,
                device=DEVICE,
                top_ks=EVAL100_TOP_KS,
                prefix="eval100",
            )
            if eval100_metrics is not None:
                log_dict.update(eval100_metrics)

        if mixed_eval_queries is not None and mixed_eval_gallery_entries is not None and mixed_eval_gallery_x is not None:
            mixed_eval_gallery_embs = lora_proj(lora_enc(mixed_eval_gallery_x.to(DEVICE, non_blocking=True))).detach()
            mixed_eval_gallery_ids = [f"{entry_type}:{entry_id}" for entry_type, entry_id in mixed_eval_gallery_entries]
            mixed_eval_metrics = evaluate_multi_positive_retrieval(
                queries=mixed_eval_queries,
                gallery_lora_ids=mixed_eval_gallery_ids,
                gallery_lora_embs=mixed_eval_gallery_embs,
                image_enc=image_enc,
                image_proj=image_proj,
                device=DEVICE,
                top_ks=MIXED_EVAL_TOP_KS,
                prefix="mixed_eval60",
            )
            if mixed_eval_metrics is not None:
                log_dict.update(mixed_eval_metrics)

    wandb.log(log_dict)

    is_best = avg_loss < best_train_loss
    if is_best:
        best_train_loss = avg_loss

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

    message = (
        f"[epoch {epoch + 1:02d}] "
        f"train_loss={avg_loss:.6f} "
        f"best={best_train_loss:.6f} "
        f"mixed_rank_active={int(mixed_rank_active)} "
        f"time={epoch_time_sec:.1f}s "
        f"throughput={train_samples_per_sec:.1f}/s"
    )
    metric_parts = []
    if eval100_metrics is not None:
        for k in EVAL100_TOP_KS:
            key = f"eval100_hit@{k}"
            if key in eval100_metrics:
                metric_parts.append(f"{key}={eval100_metrics[key]:.4f}")
        if "eval100_median_rank" in eval100_metrics:
            metric_parts.append(f"eval100_median_rank={eval100_metrics['eval100_median_rank']:.2f}")
    if mixed_eval_metrics is not None:
        for k in MIXED_EVAL_TOP_KS:
            key = f"mixed_eval60_hit@{k}"
            if key in mixed_eval_metrics:
                metric_parts.append(f"{key}={mixed_eval_metrics[key]:.4f}")
    if metric_parts:
        message += " " + " ".join(metric_parts)
    print(message)

wandb.finish()

import torch
from PIL import Image


def _load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


def collate_image_lora(batch, model_matrix_dict, image_preprocess=None):
    """
    batch: List[Tuple[image_path(str), lora_id(str), positive_lora_ids(List[str])]]
    returns:
      image_inputs: List[str] or torch.Tensor [B, C, H, W]
      lora_x: torch.Tensor [B, T, D]
      lora_ids: List[str]
      positive_mask: torch.BoolTensor [B, B]
    """
    image_paths = [b[0] for b in batch]
    lora_ids = [b[1] for b in batch]
    positive_lora_ids_rows = [b[2] if len(b) > 2 else [b[1]] for b in batch]

    mats = []
    for lora_id in lora_ids:
        if lora_id not in model_matrix_dict:
            raise KeyError(f"lora_id '{lora_id}' not found in model_matrix_dict")
        mats.append(model_matrix_dict[lora_id])

    lora_x = torch.stack(mats, dim=0)

    if image_preprocess is None:
        image_inputs = image_paths
    else:
        images = [image_preprocess(_load_rgb_image(path)) for path in image_paths]
        image_inputs = torch.stack(images, dim=0)

    positive_mask = torch.tensor(
        [
            [lora_id in positive_lora_ids for lora_id in lora_ids]
            for positive_lora_ids in positive_lora_ids_rows
        ],
        dtype=torch.bool,
    )

    return image_inputs, lora_x, lora_ids, positive_mask


def collate_image_mixed_relations(batch, image_preprocess=None):
    """
    batch: List[Dict[str, Any]]
    returns:
      image_inputs: torch.Tensor [B, C, H, W] or List[str]
      base_lora_ids: List[str]
      source_types: List[str]
      related_mixed_lora_ids: List[List[str]]
      base_related_mixed_lora_ids: List[List[str]]
    """
    image_paths = [item["image_path"] for item in batch]
    base_lora_ids = [str(item["base_lora_id"]) for item in batch]
    source_types = [str(item["source_type"]) for item in batch]
    related_mixed_lora_ids = [
        [str(x) for x in item["related_mixed_lora_ids"]]
        for item in batch
    ]
    base_related_mixed_lora_ids = [
        [str(x) for x in item["base_related_mixed_lora_ids"]]
        for item in batch
    ]

    if image_preprocess is None:
        image_inputs = image_paths
    else:
        images = [image_preprocess(_load_rgb_image(path)) for path in image_paths]
        image_inputs = torch.stack(images, dim=0)

    return image_inputs, base_lora_ids, source_types, related_mixed_lora_ids, base_related_mixed_lora_ids

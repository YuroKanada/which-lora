import json
from pathlib import Path

from torch.utils.data import Dataset


class ImageMixedRelationsDataset(Dataset):
    """
    jsonl: 1 line = {
      "image_path": str,
      "base_lora_id": str,
      "source_type": str,
      "related_mixed_lora_ids": List[str],
      "base_related_mixed_lora_ids": List[str],
      ...
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        image_path_key: str = "image_path",
        base_lora_id_key: str = "base_lora_id",
        source_type_key: str = "source_type",
        related_mixed_ids_key: str = "related_mixed_lora_ids",
        base_related_mixed_ids_key: str = "base_related_mixed_lora_ids",
        image_root_dir: str | None = None,
        image_root_dirs: list[str] | None = None,
        path_replacements: list[tuple[str, str]] | None = None,
    ):
        self.image_path_key = image_path_key
        self.base_lora_id_key = base_lora_id_key
        self.source_type_key = source_type_key
        self.related_mixed_ids_key = related_mixed_ids_key
        self.base_related_mixed_ids_key = base_related_mixed_ids_key
        self.image_root_dir = Path(image_root_dir).expanduser().resolve() if image_root_dir else None
        self.image_root_dirs = [
            Path(root).expanduser().resolve()
            for root in (image_root_dirs or [])
        ]
        self.path_replacements = [(str(src), str(dst)) for src, dst in (path_replacements or [])]
        self.items = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                required_keys = {
                    self.image_path_key,
                    self.base_lora_id_key,
                    self.source_type_key,
                    self.related_mixed_ids_key,
                    self.base_related_mixed_ids_key,
                }
                missing = [key for key in required_keys if key not in item]
                if missing:
                    raise ValueError(f"Invalid item (missing keys={missing}): {item}")

                image_path = self._resolve_image_path(item[self.image_path_key])
                related_ids = [str(x) for x in item[self.related_mixed_ids_key]]
                base_related_ids = [str(x) for x in item[self.base_related_mixed_ids_key]]
                self.items.append(
                    {
                        "image_path": image_path,
                        "base_lora_id": str(item[self.base_lora_id_key]),
                        "source_type": str(item[self.source_type_key]),
                        "related_mixed_lora_ids": related_ids,
                        "base_related_mixed_lora_ids": base_related_ids,
                    }
                )

        if not self.items:
            raise ValueError(f"No items loaded from {jsonl_path}")

    def _resolve_image_path(self, image_path_value: str) -> str:
        image_path = Path(str(image_path_value)).expanduser()
        image_path_str = str(image_path)

        for src_prefix, dst_prefix in self.path_replacements:
            if image_path_str.startswith(src_prefix):
                image_path = Path(image_path_str.replace(src_prefix, dst_prefix, 1)).expanduser()
                image_path_str = str(image_path)
                break

        if image_path.is_absolute():
            return str(image_path)

        for root_dir in self.image_root_dirs:
            candidate = (root_dir / image_path).resolve()
            if candidate.exists():
                return str(candidate)

        if self.image_root_dir is not None:
            candidate = (self.image_root_dir / image_path).resolve()
            if candidate.exists():
                return str(candidate)
            return str(candidate)
        return str(image_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

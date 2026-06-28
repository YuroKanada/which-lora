import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

from torch.utils.data import Dataset


class ImageLoRAPairsDataset(Dataset):
    """
    jsonl: 1 line = {"image_path": str, "lora_id": str, ...}
    """

    def __init__(
        self,
        jsonl_path: str,
        image_path_key: str = "image_path",
        lora_id_key: str = "lora_id",
        image_root_dir: str | None = None,
        path_replacements: list[tuple[str, str]] | None = None,
        base_plan_csv_path: str | None = None,
        mixed_plan_csv_path: str | None = None,
        include_base_loras: bool = True,
        include_mixed_loras: bool = True,
    ):
        self.image_path_key = image_path_key
        self.lora_id_key = lora_id_key
        self.image_root_dir = Path(image_root_dir).expanduser().resolve() if image_root_dir else None
        self.path_replacements = [(str(src), str(dst)) for src, dst in (path_replacements or [])]
        self.source_image_filenames_by_lora_id = self._load_source_image_filenames(
            base_plan_csv_path=base_plan_csv_path,
            mixed_plan_csv_path=mixed_plan_csv_path,
        )
        self.items = []
        source_image_to_lora_ids = defaultdict(set)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if self.image_path_key not in item or self.lora_id_key not in item:
                    raise ValueError(f"Invalid item (missing keys): {item}")
                image_path = self._resolve_image_path(item[self.image_path_key])
                lora_id = str(item[self.lora_id_key])
                if not self._should_include_lora_id(
                    lora_id=lora_id,
                    include_base_loras=include_base_loras,
                    include_mixed_loras=include_mixed_loras,
                ):
                    continue
                image_index = int(item.get("image_index", -1))
                source_image_filename = self._resolve_source_image_filename(
                    lora_id=lora_id,
                    image_index=image_index,
                )
                self.items.append((image_path, lora_id, source_image_filename))
                if source_image_filename is not None:
                    source_image_to_lora_ids[source_image_filename].add(lora_id)

        if not self.items:
            raise ValueError(f"No items loaded from {jsonl_path}")

        enriched_items = []
        for image_path, lora_id, source_image_filename in self.items:
            positive_lora_ids = sorted(source_image_to_lora_ids.get(source_image_filename, {lora_id}))
            enriched_items.append((image_path, lora_id, positive_lora_ids))
        self.items = enriched_items

    def _load_source_image_filenames(self, base_plan_csv_path: str | None, mixed_plan_csv_path: str | None) -> dict[str, list[str]]:
        mapping = {}
        for prefix, plan_csv_path in (
            ("base", base_plan_csv_path),
            ("mixed", mixed_plan_csv_path),
        ):
            if not plan_csv_path:
                continue
            with open(plan_csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_lora_id = str(row["lora_id"])
                    typed_lora_id = raw_lora_id if ":" in raw_lora_id else f"{prefix}:{raw_lora_id}"
                    image_filenames = ast.literal_eval(row.get("image_filenames", "[]"))
                    mapping[typed_lora_id] = [str(x) for x in image_filenames]
        return mapping

    def _resolve_source_image_filename(self, lora_id: str, image_index: int) -> str | None:
        filenames = self.source_image_filenames_by_lora_id.get(lora_id)
        if filenames is None or not (0 <= image_index < len(filenames)):
            return None
        return filenames[image_index]

    def _should_include_lora_id(
        self,
        lora_id: str,
        include_base_loras: bool,
        include_mixed_loras: bool,
    ) -> bool:
        text = str(lora_id)
        if text.startswith("mixed:"):
            return include_mixed_loras
        return include_base_loras

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
        if self.image_root_dir is not None:
            return str((self.image_root_dir / image_path).resolve())
        return str(image_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

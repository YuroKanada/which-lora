import os, json, numpy as np, torch
from collections import defaultdict


def _format_model_id(raw_model_id, namespace=None, add_lora_prefix=False, auto_base_namespace_for_digits=False):
    model_id = str(raw_model_id)
    if ":" in model_id:
        return model_id
    if auto_base_namespace_for_digits and model_id.isdigit():
        return f"base:lora_{model_id.zfill(6)}"
    if add_lora_prefix and not model_id.startswith("lora_"):
        model_id = f"lora_{model_id}"
    if namespace:
        model_id = f"{namespace}:{model_id}"
    return model_id


def load_model_matrix(category_dir, dimension, namespace=None, add_lora_prefix=False):
    files = sorted([f for f in os.listdir(category_dir) if f.endswith(".npz")])
    category_names = [f.replace("compressed_", "").replace(f"_" + str(dimension) + "d.npz", "") for f in files]
    has_namespaced_ids = False
    for fname in files:
        data = np.load(os.path.join(category_dir, fname), allow_pickle=True)
        mids = data["model_ids"]
        if any(":" in str(mid) for mid in mids[: min(16, len(mids))]):
            has_namespaced_ids = True
            break
    temp = defaultdict(dict)
    for fname, cat in zip(files, category_names):
        data = np.load(os.path.join(category_dir, fname))
        for mid, vec in zip(data["model_ids"], data["vectors"]):
            formatted_mid = _format_model_id(
                mid,
                namespace=namespace,
                add_lora_prefix=add_lora_prefix,
                auto_base_namespace_for_digits=(namespace is None and has_namespaced_ids),
            )
            temp[formatted_mid][cat] = vec

    if not temp:
        print("Loaded 0 model matrices")
        return {}

    common_categories = [
        cat for cat in category_names
        if all(cat in temp[mid] for mid in temp)
    ]
    if not common_categories:
        print("Loaded 0 model matrices")
        return {}

    model_matrix_dict = {
        mid: torch.tensor(np.stack([temp[mid][c] for c in common_categories]), dtype=torch.float32)
        for mid in temp
    }
    ignored_categories = len(category_names) - len(common_categories)
    if ignored_categories > 0:
        print(
            f"Loaded {len(model_matrix_dict)} model matrices "
            f"using {len(common_categories)}/{len(category_names)} common categories"
        )
    else:
        print(f"Loaded {len(model_matrix_dict)} model matrices")
    return model_matrix_dict


def load_triplets(dataset_dir, train_name, val_name):
    def _load(fname):
        with open(os.path.join(dataset_dir, fname)) as f:
            return [json.loads(l) for l in f]
    return _load(train_name), _load(val_name)

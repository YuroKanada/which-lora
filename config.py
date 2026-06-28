import torch


CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dimension": 128,
    "epochs": 15,
    "batch_size": 128,
    "weight_decay": 1e-4,
    "lr": 1e-4,
    "num_workers": 4,
    "grad_clip_norm": 0.0,

    # LoRA encoder
    "lora_encoder_mode": "full",  # "full" | "compress_only"
    "num_layers": 4,
    "num_heads": 4,
    "ff_dim": 256,
    "lora_aggregator_type": "mlp",  # "mlp" | "mean"
    "lora_aggregator_hidden_dim": 64,
    # Image encoder
    "image_encoder_type": "openclip",
    "image_encoder_model_name": "ViT-B-32",
    "image_encoder_pretrained": "laion2b_s34b_b79k",
    "freeze_image": True,
    "freeze_image_proj": False,
    "train_image_projection": False,
    "image_proj_type": "linear",
    "image_proj_hidden_dim": 1024,
    "image_proj_dropout": 0.0,

    # LoRA projection / pretrained
    "use_lora_proj": True,
    "freeze_lora_encoder": False,
    "freeze_lora_proj": False,
    "lora_proj_type": "linear",
    "lora_proj_hidden_dim": 256,
    "lora_proj_dropout": 0.0,
    "pretrained_lora_encoder": False,
    "pretrained_encoder_path": "/home/kanada/my-jupyterlab/work-dir/triplet_transformer+aggregator/saved_models/encoder_epoch15_202602041312.pt",
    "pretrained_aggregator_path": "/home/kanada/my-jupyterlab/work-dir/triplet_transformer+aggregator/saved_models/aggregator_epoch15_202602041312.pt",

    "infonce_temperature": 0.07,

    # Learning-to-rank loss
    "use_rank_loss": False,
    "rank_loss_weight": 0.3,
    "rank_loss_margin": 0.2,
    "rank_topk": 3, #正例以外で上位何件までを負例として考慮するか
    "use_mixed_rank_loss": False,
    "mixed_rank_loss_weight": 0.25,
    "mixed_rank_loss_margin": 0.1,
    "mixed_rank_topk": 3,
    "mixed_rank_warmup_epochs": 20,
    "use_mix_ratio_geometry_loss": False,
    "mix_ratio_geometry_loss_weight": 0.1,
    "mix_ratio_geometry_loss_margin": 0.0,
    "mix_ratio_geometry_warmup_epochs": 30,

    # Dataset
    "train_pairs_jsonl": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/metadata/image_lora_pairs_base400_plus_mixed90_namespace_one_image_per_line.jsonl",
    "include_base_train_pairs": True,
    "include_mixed_train_pairs": True,
    "base_plan_csv": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32/metadata/lora_plan.csv",
    "base_lora_namespace": None,
    "mixed_lora_namespace": None,
    "mixed_train_relations_jsonl": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/metadata/image_mixed_relations_train_namespace.jsonl",
    "mixed_train_plan_csv": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/metadata/lora_plan.csv",
    "image_root_dir": "/home/kanada/my-jupyterlab/work-dir",
    "mixed_image_root_dirs": [
        "/home/kanada/my-jupyterlab/work-dir/wikiart",
        "/home/kanada/my-jupyterlab/work-dir",
    ],
    "image_path_replacements": [
        ("/root/work-dir", "/home/kanada/my-jupyterlab/work-dir"),
    ],
    "validation_eval_every": 1,
    "validation_batch_size": 64,
    "eval100_pairs_jsonl": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32_eval_tmp/metadata/eval100_pairs_namespace_one_image_per_line.jsonl",
    "mixed_eval_queries_jsonl": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/metadata/mixed_query_images_10donors_x10.jsonl",
    "mixed_eval_ids_json": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/metadata/mixed_eval_ids_namespace.json",
    "mixed_eval_plan_csv": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/metadata/lora_plan.csv",

    # Paths
    "category_dir": "/home/kanada/my-jupyterlab/work-dir/preprocessing_LoRA/compressed_rank32_artLoRA_train",
    "eval100_category_dir": "/home/kanada/my-jupyterlab/work-dir/preprocessing_LoRA/compressed_rank32_artLoRA_eval100",
    "mixed_eval_category_dir": "/home/kanada/my-jupyterlab/work-dir/preprocessing_LoRA/compressed_rank32_artLoRA_mixed_eval60",
    "save_dir": "/home/kanada/my-jupyterlab/work-dir/Image_LoRA_encoder/saved_models",

    # Logging
    "wandb_enabled": True,
    "wandb_mode": "online",
    "wandb_project": "image_lora_Retrieval_detection",
    "wandb_run_name_prefix": "run",
    "wandb_api_key": None,
}

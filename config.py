import torch


CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dimension": 128,
    "epochs": 50,
    "batch_size": 64,
    "weight_decay": 1e-4,
    "lr": 1e-4,
    "num_workers": 4,
    "grad_clip_norm": 0.0,

    # LoRA encoder
    "num_layers": 4,
    "num_heads": 4,
    "ff_dim": 256,
    "max_len": 264,
    "use_positional_embedding": True,
    "use_mlp_aggregator": True,
    "aggregator_hidden_dim": 64,

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
    "image_proj_normalize": True,

    # LoRA projection / pretrained
    "freeze_lora_encoder": False,
    "freeze_lora_proj": False,
    "lora_proj_type": "linear",
    "lora_proj_hidden_dim": 256,
    "lora_proj_dropout": 0.0,
    "pretrained_lora_encoder": False,
    "pretrained_encoder_path": "/home/kanada/my-jupyterlab/work-dir/triplet_transformer+aggregator/saved_models/encoder_epoch15_202602041312.pt",
    "pretrained_aggregator_path": "/home/kanada/my-jupyterlab/work-dir/triplet_transformer+aggregator/saved_models/aggregator_epoch15_202602041312.pt",
    "pretrained_strict": True,

    "infonce_temperature": 0.07,

    # Dataset
    "train_pairs_jsonl": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32/metadata/image_lora_pairs_artist100_rank32_one_image_per_line.jsonl",
    "validation_pairs_jsonl": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32/metadata/validation_image_lora_pairs_unused30.jsonl",
    "image_path_key": "image_path",
    "lora_id_key": "lora_id",
    "image_root_dir": "/home/kanada/my-jupyterlab/work-dir",
    "image_path_replacements": [
        ("/root/work-dir", "/home/kanada/my-jupyterlab/work-dir"),
    ],
    "validation_eval_every": 1,
    "validation_batch_size": 64,
    "validation_top_ks": [1, 5, 10],

    # Paths
    "category_dir": "/home/kanada/my-jupyterlab/work-dir/preprocessing_LoRA/compressed_rank32_artLoRA",
    "dataset_dir": "/home/kanada/my-jupyterlab/work-dir/WikiArt_LoRA_Dataset_artist100_rank32",
    "save_dir": "/home/kanada/my-jupyterlab/work-dir/Image_LoRA_encoder/saved_models",
    "checkpoint_dir": "/home/kanada/my-jupyterlab/work-dir/Image_LoRA_encoder/checkpoints",

    # Logging
    "wandb_project": "image_lora_retrieval",
    "wandb_run_name_prefix": "run",
    "wandb_api_key": None,
}

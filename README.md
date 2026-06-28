# Retrieval of LoRA Models based on Layer-Wise Weight Embedding without Metadata

This repository implements a framework for learning embedding representations of LoRA models directly from their adapter weights, without relying on textual metadata. The learned representations are aligned with image features so that each LoRA can be retrieved from example images and analyzed in a shared embedding space.

The current codebase supports training with image-LoRA pairs, evaluation on base-LoRA retrieval, and mixed-LoRA relation learning.

<!-- ## Publication

This work has been accepted to ICMR 2026:

Kanada, Yuro, Yuma Oe, Huu-Long Pham, Makoto P. Kato, Hiroaki Ohshima, Sumio Fujita, and Yoshiyuki Shoji. "Retrieval of LoRA Models based on Layer-Wise Weight Embedding without Metadata." In *Proceedings of the 16th ACM International Conference on Multimedia Retrieval (ICMR 2026)*, to appear, 2026. -->

<!-- ## Citation

If you use this repository or the encoder in your research, please cite:

```bibtex
@inproceedings{kanada2026retrieval,
  title = {Retrieval of LoRA Models based on Layer-Wise Weight Embedding without Metadata},
  author = {Kanada, Yuro and Oe, Yuma and Pham, Huu-Long and Kato, Makoto P. and Ohshima, Hiroaki and Fujita, Sumio and Shoji, Yoshiyuki},
  booktitle = {Proceedings of the 16th ACM International Conference on Multimedia Retrieval},
  year = {2026},
  note = {To appear}
}
``` -->

## Repository Structure

```text
Image_LoRA_encoder/
├── README.md                   # Project overview
├── config.py                   # Global configuration, dataset paths, and training options
├── main_train.py               # Training, validation, and retrieval evaluation entry point
├── dataset/
│   ├── collate.py              # Batch collation for base and mixed training samples
│   ├── image_lora_pairs.py     # Dataset for image-LoRA pair supervision
│   ├── image_mixed_relations.py# Dataset for mixed-LoRA relation supervision
│   └── loader.py               # LoRA weight loading and matrix preparation
├── model/
│   ├── aggregator.py           # Token aggregation modules for LoRA sequences
│   ├── heads.py                # Projection heads for image and LoRA embeddings
│   ├── image_encoder_factory.py# Image encoder builder
│   ├── lora_encoder.py         # LoRA encoder wrapper
│   ├── openclip_image_encoder.py # OpenCLIP-based image encoder
│   └── transformer_encoder.py  # Transformer encoder for layer-wise LoRA weights
├── train/
│   ├── infonce.py              # InfoNCE loss
│   ├── mixed_losses.py         # Mixed-LoRA auxiliary losses
│   └── ranking.py              # Ranking-based retrieval loss
└── utils/
    ├── eval_significance.py    # Statistical significance utilities for evaluation
    └── helper.py               # Checkpoint, gradient, and training helpers
```

## Training

Set the dataset paths and output paths in [`config.py`](/home/kanada/my-jupyterlab/work-dir/Image_LoRA_encoder/config.py), then run:

```bash
python3 main_train.py
```

## Notes

- `config.py` currently contains absolute local paths for training data, evaluation data, and checkpoints.
- Weights and experiment artifacts such as `saved_models/`, `wandb/`, generated outputs, and notebooks are intentionally kept out of version control.

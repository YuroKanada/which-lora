# Which LoRA Was Trained on My Image? <br>Retrieving Candidate LoRAs from Internal Weights</br>

This repository implements a framework for learning embedding representations of LoRA models directly from their adapter weights, without relying on textual metadata. 
The learned representations are aligned with image features so that each LoRA can be retrieved from example images and analyzed in a shared embedding space.

The current codebase supports training with image-LoRA pairs, evaluation on base-LoRA retrieval, and mixed-LoRA relation learning.

## Publication

This paper has been accepted as a full paper at the International Conference on Multimedia Modeling (MMM 2027).

Yuro Kanada and Yoshiyuki Shoji. "Which LoRA Was Trained on My Image? Retrieving Candidate LoRAs from Internal Weights." In *Proceedings of the International Conference on Multimedia Modeling (MMM 2027)*, to appear, 2027.

## Citation

If you use this repository or the encoder in your research, please cite:

```bibtex
@inproceedings{kanada2027which,
  title = {Which {LoRA} Was Trained on My Image? Retrieving Candidate {LoRA}s from Internal Weights},
  author = {Kanada, Yuro and Shoji, Yoshiyuki},
  booktitle = {Proceedings of the International Conference on Multimedia Modeling},
  year = {2027},
  note = {To appear}
}
```

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

## Generation-Based Retrieval Baseline (GBR)

Generation-Based Retrieval (GBR) is a comparison method used in the paper, not
the proposed internal-weight retrieval method. For each candidate LoRA, GBR
generates ten images with Stable Diffusion 1.5, extracts their image features,
and averages the features to construct the LoRA's retrieval representation.
Candidate LoRAs are then ranked according to the similarity between this
representation and the query image feature. The generation code is provided in
`generate_sd15_clip_baseline_images.py`.

Some code and saved evaluation artifacts use `GBL` or `gbl` as a legacy
internal label. These names refer to the same GBR comparison method.

The following prompts were used for every LoRA:

1. `a portrait of a person, detailed painting`
2. `a landscape with mountains and trees, detailed painting`
3. `a city street at night, detailed painting`
4. `a still life with flowers and objects, detailed painting`
5. `an interior room with a window, detailed painting`
6. `a seascape with boats and waves, detailed painting`
7. `a garden with blooming flowers, detailed painting`
8. `a village street with small houses, detailed painting`
9. `a person sitting at a table, detailed painting`
10. `a forest path with sunlight, detailed painting`

The negative prompt was:

```text
worst quality, low quality, blurry, distorted, deformed, bad anatomy, extra fingers, watermark, text, signature
```

## Notes

- `config.py` currently contains absolute local paths for training data, evaluation data, and checkpoints.
- Weights and experiment artifacts such as `saved_models/`, `wandb/`, generated outputs, and notebooks are intentionally kept out of version control.

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline


GEN_PROMPTS = [
    "a portrait of a person, detailed painting",
    "a landscape with mountains and trees, detailed painting",
    "a city street at night, detailed painting",
    "a still life with flowers and objects, detailed painting",
    "an interior room with a window, detailed painting",
    "a seascape with boats and waves, detailed painting",
    "a garden with blooming flowers, detailed painting",
    "a village street with small houses, detailed painting",
    "a person sitting at a table, detailed painting",
    "a forest path with sunlight, detailed painting",
]

NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, distorted, deformed, bad anatomy, "
    "extra fingers, watermark, text, signature"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SD1.5 baseline images for eval100 and mixed-eval LoRAs."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("/home/kanada/my-jupyterlab/work-dir/v1-5-pruned-emaonly.safetensors"),
        help="Path to the SD1.5 base model safetensors/ckpt file.",
    )
    parser.add_argument(
        "--eval-lora-dir",
        type=Path,
        default=Path(
            "/home/kanada/my-jupyterlab/work-dir/"
            "WikiArt_LoRA_Dataset_artist100_rank32_eval_tmp/loras"
        ),
    )
    parser.add_argument(
        "--mixed-lora-dir",
        type=Path,
        default=Path(
            "/home/kanada/my-jupyterlab/work-dir/"
            "WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/loras"
        ),
    )
    parser.add_argument(
        "--mixed-summary-csv",
        type=Path,
        default=Path(
            "/home/kanada/my-jupyterlab/work-dir/"
            "WikiArt_LoRA_Dataset_artist100_rank32_mixed_multidonor_tmp/metadata/"
            "mixed_train_eval_split_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/kanada/my-jupyterlab/work-dir/Image_LoRA_encoder/"
            "generated_sd15_clip_baseline"
        ),
    )
    parser.add_argument(
        "--task",
        choices=["both", "eval100", "mixed_eval"],
        default="both",
        help="Which LoRA groups to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging. Applies after task filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if the output file already exists.",
    )
    return parser.parse_args()


def resolve_lora_jobs(args: argparse.Namespace) -> list[dict]:
    jobs: list[dict] = []

    if args.task in {"both", "eval100"}:
        for lora_path in sorted(args.eval_lora_dir.glob("*.safetensors")):
            jobs.append(
                {
                    "group": "eval100",
                    "lora_id": lora_path.stem,
                    "lora_path": lora_path,
                }
            )

    if args.task in {"both", "mixed_eval"}:
        mixed_df = pd.read_csv(args.mixed_summary_csv)
        mixed_eval_ids = mixed_df.loc[mixed_df["split"] == "eval", "lora_id"].tolist()
        for lora_id in mixed_eval_ids:
            lora_path = args.mixed_lora_dir / f"{lora_id}.safetensors"
            jobs.append(
                {
                    "group": "mixed_eval",
                    "lora_id": lora_id,
                    "lora_path": lora_path,
                }
            )

    if args.limit is not None:
        jobs = jobs[: args.limit]

    missing = [job for job in jobs if not job["lora_path"].exists()]
    if missing:
        preview = ", ".join(job["lora_id"] for job in missing[:10])
        raise FileNotFoundError(f"Missing LoRA files: {preview}")

    return jobs


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def canonical_lora_key(group: str, lora_id: str) -> str:
    return f"{group}__{lora_id}"


def output_path(output_dir: Path, group: str, lora_id: str, prompt_index: int) -> Path:
    return output_dir / canonical_lora_key(group, lora_id) / f"prompt_{prompt_index:02d}.png"


def build_records(jobs: Iterable[dict], output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for job in jobs:
        for prompt_index, prompt in enumerate(GEN_PROMPTS):
            records.append(
                {
                    "group": job["group"],
                    "lora_id": job["lora_id"],
                    "canonical_lora_key": canonical_lora_key(job["group"], job["lora_id"]),
                    "lora_path": str(job["lora_path"]),
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "output_path": str(
                        output_path(output_dir, job["group"], job["lora_id"], prompt_index)
                    ),
                }
            )
    return records


def save_manifest(args: argparse.Namespace, jobs: list[dict]) -> None:
    manifest_dir = args.output_dir / "metadata"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(jobs, args.output_dir)
    csv_path = manifest_dir / "generation_manifest.csv"
    json_path = manifest_dir / "generation_config.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "lora_id",
                "canonical_lora_key",
                "lora_path",
                "prompt_index",
                "prompt",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    config = {
        "base_model": str(args.base_model),
        "task": args.task,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "lora_scale": args.lora_scale,
        "width": args.width,
        "height": args.height,
        "negative_prompt": NEGATIVE_PROMPT,
        "prompts": GEN_PROMPTS,
        "num_loras": len(jobs),
        "num_images": len(records),
        "canonical_key_format": "{group}__{lora_id}",
    }
    json_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def save_timing_outputs(
    output_dir: Path,
    timing_rows: list[dict],
    lora_summary_rows: list[dict],
    summary: dict,
) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    timing_csv = metadata_dir / "generation_timing.csv"
    lora_summary_csv = metadata_dir / "generation_lora_summary.csv"
    summary_json = metadata_dir / "generation_timing_summary.json"

    with timing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "lora_id",
                "canonical_lora_key",
                "prompt_index",
                "output_path",
                "status",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    with lora_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "lora_id",
                "canonical_lora_key",
                "lora_load_seconds",
                "num_generated_images",
                "num_skipped_images",
                "generated_images_seconds",
                "total_job_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(lora_summary_rows)

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pipeline(base_model: Path) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_single_file(
        str(base_model),
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def unload_lora_if_needed(pipe: StableDiffusionPipeline) -> None:
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass


def generate_for_job(
    pipe: StableDiffusionPipeline,
    job: dict,
    args: argparse.Namespace,
    timing_rows: list[dict],
) -> dict:
    unload_lora_if_needed(pipe)
    load_started_at = time.perf_counter()
    pipe.load_lora_weights(str(job["lora_path"]))
    lora_load_seconds = time.perf_counter() - load_started_at

    generated_count = 0
    skipped_count = 0
    generated_seconds = 0.0

    for prompt_index, prompt in enumerate(GEN_PROMPTS):
        save_path = output_path(args.output_dir, job["group"], job["lora_id"], prompt_index)
        if save_path.exists() and not args.overwrite:
            skipped_count += 1
            timing_rows.append(
                {
                    "group": job["group"],
                    "lora_id": job["lora_id"],
                    "canonical_lora_key": canonical_lora_key(job["group"], job["lora_id"]),
                    "prompt_index": prompt_index,
                    "output_path": str(save_path),
                    "status": "skipped_existing",
                    "elapsed_seconds": 0.0,
                }
            )
            continue

        ensure_parent(save_path)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        started_at = time.perf_counter()
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            cross_attention_kwargs={"scale": args.lora_scale},
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]
        image.save(save_path)
        elapsed = time.perf_counter() - started_at
        generated_count += 1
        generated_seconds += elapsed
        timing_rows.append(
            {
                "group": job["group"],
                "lora_id": job["lora_id"],
                "canonical_lora_key": canonical_lora_key(job["group"], job["lora_id"]),
                "prompt_index": prompt_index,
                "output_path": str(save_path),
                "status": "generated",
                "elapsed_seconds": round(elapsed, 6),
            }
        )
        print(f"  prompt_{prompt_index:02d}: {elapsed:.2f}s")

    total_job_seconds = lora_load_seconds + generated_seconds
    print(
        f"  lora_load: {lora_load_seconds:.2f}s, "
        f"generate_total: {generated_seconds:.2f}s, job_total: {total_job_seconds:.2f}s"
    )
    return {
        "group": job["group"],
        "lora_id": job["lora_id"],
        "canonical_lora_key": canonical_lora_key(job["group"], job["lora_id"]),
        "lora_load_seconds": round(lora_load_seconds, 6),
        "num_generated_images": generated_count,
        "num_skipped_images": skipped_count,
        "generated_images_seconds": round(generated_seconds, 6),
        "total_job_seconds": round(total_job_seconds, 6),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this generation script.")
    if not args.base_model.exists():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    jobs = resolve_lora_jobs(args)
    save_manifest(args, jobs)

    overall_started_at = time.perf_counter()
    pipe = load_pipeline(args.base_model)
    completed = 0
    timing_rows: list[dict] = []
    lora_summary_rows: list[dict] = []
    try:
        for index, job in enumerate(jobs, start=1):
            print(
                f"[{index}/{len(jobs)}] generating group={job['group']} "
                f"lora_id={job['lora_id']}"
            )
            job_summary = generate_for_job(pipe, job, args, timing_rows)
            lora_summary_rows.append(job_summary)
            completed += 1
    finally:
        del pipe
        torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - overall_started_at
    generated_rows = [row for row in timing_rows if row["status"] == "generated"]
    summary = {
        "num_loras_completed": completed,
        "num_loras_total": len(jobs),
        "num_images_generated": len(generated_rows),
        "num_images_recorded": len(timing_rows),
        "total_elapsed_seconds": round(total_elapsed, 6),
        "mean_generated_image_seconds": round(
            sum(row["elapsed_seconds"] for row in generated_rows) / len(generated_rows), 6
        ) if generated_rows else 0.0,
    }
    save_timing_outputs(args.output_dir, timing_rows, lora_summary_rows, summary)

    print(f"Completed LoRAs: {completed}/{len(jobs)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Total elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()

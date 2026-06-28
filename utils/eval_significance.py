from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest


DEFAULT_TOP_KS = (1, 5, 10, 20)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def attach_rank_metrics(
    df: pd.DataFrame,
    rank_col: str,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    copy: bool = True,
) -> pd.DataFrame:
    out = df.copy() if copy else df
    out["rank"] = out[rank_col].astype(int)
    out["reciprocal_rank"] = 1.0 / out["rank"].astype(float)
    for k in top_ks:
        hit_col = f"hit@{k}"
        if hit_col not in out.columns:
            out[hit_col] = (out["rank"] <= int(k)).astype(int)
    return out


def summarize_query_metrics(
    df: pd.DataFrame,
    rank_col: str = "rank",
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
) -> dict[str, float]:
    rank = df[rank_col].astype(float)
    metrics = {
        "num_queries": int(len(df)),
        "mean_rank": float(rank.mean()) if len(df) else math.nan,
        "median_rank": float(rank.median()) if len(df) else math.nan,
        "mrr": float((1.0 / rank).mean()) if len(df) else math.nan,
    }
    for k in top_ks:
        hit_col = f"hit@{k}"
        if hit_col in df.columns:
            metrics[hit_col] = float(df[hit_col].mean()) if len(df) else math.nan
        recall_col = f"recall@{k}"
        if recall_col in df.columns:
            metrics[recall_col] = float(df[recall_col].mean()) if len(df) else math.nan
    return metrics


def summarize_group_macro(
    df: pd.DataFrame,
    group_col: str,
    rank_col: str = "rank",
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    for group_value, group_df in df.groupby(group_col, sort=True):
        row = {group_col: group_value}
        row.update(summarize_query_metrics(group_df, rank_col=rank_col, top_ks=top_ks))
        rows.append(row)
    group_df = pd.DataFrame(rows)
    if len(group_df) == 0:
        return group_df, {}
    macro = group_df.drop(columns=[group_col]).mean(numeric_only=True).to_dict()
    macro = {key: float(value) for key, value in macro.items()}
    return group_df, macro


def save_eval_artifacts(
    output_dir: str | Path,
    details_df: pd.DataFrame,
    rank_col: str,
    group_col: str,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    metadata: dict | None = None,
    details_filename: str = "details.csv",
    macro_filename: str = "group_macro.csv",
    summary_filename: str = "summary.json",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details_out = attach_rank_metrics(details_df, rank_col=rank_col, top_ks=top_ks, copy=True)
    details_path = output_dir / details_filename
    macro_path = output_dir / macro_filename
    summary_path = output_dir / summary_filename

    details_out.to_csv(details_path, index=False)
    group_df, macro = summarize_group_macro(details_out, group_col=group_col, rank_col="rank", top_ks=top_ks)
    group_df.to_csv(macro_path, index=False)

    payload = {
        "metadata": metadata or {},
        "rank_col": rank_col,
        "group_col": group_col,
        "top_ks": [int(k) for k in top_ks],
        "micro": summarize_query_metrics(details_out, rank_col="rank", top_ks=top_ks),
        "macro": macro,
        "details_csv": str(details_path),
        "group_macro_csv": str(macro_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def infer_rank_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    candidates = [preferred, "rank", "best_positive_rank", "expected_rank"]
    for candidate in candidates:
        if candidate and candidate in df.columns:
            return candidate
    raise ValueError("Could not infer rank column. Tried: rank, best_positive_rank, expected_rank")


def normalize_detail_columns(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    out = df.copy() if copy else df
    alias_pairs = [
        ("query_image_path", "image_path"),
        ("image_path", "query_image_path"),
    ]
    for target, source in alias_pairs:
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    return out


def load_saved_details(path: str | Path, rank_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_detail_columns(df, copy=False)
    inferred_rank_col = infer_rank_col(df, preferred=rank_col)
    return attach_rank_metrics(df, rank_col=inferred_rank_col, top_ks=DEFAULT_TOP_KS, copy=True)


def _paired_bootstrap_ci(
    diffs: np.ndarray,
    num_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if len(diffs) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(num_bootstrap, len(diffs)))
    samples = diffs[indices].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _paired_permutation_pvalue(
    diffs: np.ndarray,
    num_resamples: int = 20000,
    seed: int = 0,
) -> float:
    if len(diffs) == 0:
        return math.nan
    diffs = np.asarray(diffs, dtype=float)
    observed = abs(diffs.mean())
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(num_resamples, len(diffs)))
    sampled = np.abs((signs * diffs).mean(axis=1))
    return float((np.count_nonzero(sampled >= observed) + 1) / (num_resamples + 1))


def paired_permutation_pvalue(
    left_values: Iterable[float],
    right_values: Iterable[float],
    num_resamples: int = 20000,
    seed: int = 0,
) -> float:
    left_arr = np.asarray(list(left_values), dtype=float)
    right_arr = np.asarray(list(right_values), dtype=float)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"Shape mismatch: {left_arr.shape} vs {right_arr.shape}")
    return _paired_permutation_pvalue(right_arr - left_arr, num_resamples=num_resamples, seed=seed)


def average_saved_details(
    csv_paths: Iterable[str | Path],
    join_cols: Iterable[str],
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
) -> pd.DataFrame:
    csv_paths = [Path(path) for path in csv_paths]
    if not csv_paths:
        raise ValueError("csv_paths must not be empty")

    metric_prefixes = ("rank", "reciprocal_rank", "hit@", "recall@")
    merged: pd.DataFrame | None = None

    for idx, csv_path in enumerate(csv_paths):
        df = load_saved_details(csv_path)
        available_metric_cols = [
            col
            for col in df.columns
            if col == "rank"
            or col == "reciprocal_rank"
            or col.startswith("hit@")
            or col.startswith("recall@")
        ]
        use_cols = list(join_cols) + available_metric_cols
        run_df = df[use_cols].copy()
        rename_map = {col: f"{col}__run{idx}" for col in available_metric_cols}
        run_df = run_df.rename(columns=rename_map)
        if merged is None:
            merged = run_df
        else:
            merged = merged.merge(run_df, on=list(join_cols), how="inner")

    if merged is None or len(merged) == 0:
        raise ValueError("No overlapping rows found while averaging saved details")

    result = merged[list(join_cols)].copy()
    metric_names = ["rank", "reciprocal_rank"]
    for k in top_ks:
        metric_names.append(f"hit@{k}")
        metric_names.append(f"recall@{k}")

    for metric_name in metric_names:
        run_cols = [col for col in merged.columns if col.startswith(f"{metric_name}__run")]
        if run_cols:
            result[metric_name] = merged[run_cols].mean(axis=1)

    return result


def mcnemar_exact_pvalue(left_success: np.ndarray, right_success: np.ndarray) -> tuple[float, int, int]:
    left_success = np.asarray(left_success, dtype=bool)
    right_success = np.asarray(right_success, dtype=bool)
    left_only = int(np.count_nonzero(left_success & ~right_success))
    right_only = int(np.count_nonzero(~left_success & right_success))
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0, left_only, right_only
    p_value = binomtest(min(left_only, right_only), discordant, 0.5, alternative="two-sided").pvalue
    return float(p_value), left_only, right_only


def compare_paired_details(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_cols: Iterable[str],
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    left_name: str = "left",
    right_name: str = "right",
    num_resamples: int = 20000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    join_cols = list(join_cols)
    merged = left_df.merge(
        right_df,
        on=join_cols,
        how="inner",
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )
    if len(merged) == 0:
        raise ValueError(f"No overlapping rows after merge on {join_cols}")

    results: list[dict[str, object]] = []

    reciprocal_left = merged[f"reciprocal_rank_{left_name}"].to_numpy(dtype=float)
    reciprocal_right = merged[f"reciprocal_rank_{right_name}"].to_numpy(dtype=float)
    reciprocal_diff = reciprocal_right - reciprocal_left
    rr_lo, rr_hi = _paired_bootstrap_ci(reciprocal_diff, num_bootstrap=5000, seed=seed)
    results.append(
        {
            "metric": "mrr",
            "test": "paired_permutation_on_reciprocal_rank",
            left_name: float(reciprocal_left.mean()),
            right_name: float(reciprocal_right.mean()),
            "delta_right_minus_left": float(reciprocal_diff.mean()),
            "ci95_low": rr_lo,
            "ci95_high": rr_hi,
            "p_value": _paired_permutation_pvalue(reciprocal_diff, num_resamples=num_resamples, seed=seed),
            "n": int(len(merged)),
        }
    )

    rank_left = merged[f"rank_{left_name}"].to_numpy(dtype=float)
    rank_right = merged[f"rank_{right_name}"].to_numpy(dtype=float)
    rank_diff = rank_right - rank_left
    rank_lo, rank_hi = _paired_bootstrap_ci(rank_diff, num_bootstrap=5000, seed=seed)
    results.append(
        {
            "metric": "mean_rank",
            "test": "paired_permutation_on_rank",
            left_name: float(rank_left.mean()),
            right_name: float(rank_right.mean()),
            "delta_right_minus_left": float(rank_diff.mean()),
            "ci95_low": rank_lo,
            "ci95_high": rank_hi,
            "p_value": _paired_permutation_pvalue(rank_diff, num_resamples=num_resamples, seed=seed),
            "n": int(len(merged)),
        }
    )

    for k in top_ks:
        for metric_name in (f"hit@{k}", f"recall@{k}"):
            left_col = f"{metric_name}_{left_name}"
            right_col = f"{metric_name}_{right_name}"
            if left_col not in merged.columns or right_col not in merged.columns:
                continue
            left_values = merged[left_col].to_numpy(dtype=float)
            right_values = merged[right_col].to_numpy(dtype=float)
            diffs = right_values - left_values
            ci_lo, ci_hi = _paired_bootstrap_ci(diffs, num_bootstrap=5000, seed=seed)
            row = {
                "metric": metric_name,
                left_name: float(left_values.mean()),
                right_name: float(right_values.mean()),
                "delta_right_minus_left": float(diffs.mean()),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "n": int(len(merged)),
            }
            if metric_name.startswith("hit@"):
                p_value, left_only, right_only = mcnemar_exact_pvalue(left_values > 0.5, right_values > 0.5)
                row.update(
                    {
                        "test": "mcnemar_exact",
                        "p_value": p_value,
                        f"{left_name}_only_success": left_only,
                        f"{right_name}_only_success": right_only,
                    }
                )
            else:
                row.update(
                    {
                        "test": "paired_permutation",
                        "p_value": _paired_permutation_pvalue(diffs, num_resamples=num_resamples, seed=seed),
                    }
                )
            results.append(row)

    return pd.DataFrame(results), merged


def compare_saved_csvs(
    left_csv: str | Path,
    right_csv: str | Path,
    join_cols: Iterable[str],
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
    left_name: str = "left",
    right_name: str = "right",
    left_rank_col: str | None = None,
    right_rank_col: str | None = None,
    num_resamples: int = 20000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_df = load_saved_details(left_csv, rank_col=left_rank_col)
    right_df = load_saved_details(right_csv, rank_col=right_rank_col)
    return compare_paired_details(
        left_df=left_df,
        right_df=right_df,
        join_cols=join_cols,
        top_ks=top_ks,
        left_name=left_name,
        right_name=right_name,
        num_resamples=num_resamples,
        seed=seed,
    )

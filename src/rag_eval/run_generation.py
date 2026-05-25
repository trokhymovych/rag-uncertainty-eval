from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from rag_eval.clients import GenerationRequest, make_provider
from rag_eval.columns import detect_columns, require_columns
from rag_eval.config import ensure_dir, load_config
from rag_eval.prompts import build_generation_prompt


def experiment_name(model_key: str, prompt_key: str, mode_key: str) -> str:
    return f"{model_key}__{prompt_key}__{mode_key}"


def is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan")


def run_one_experiment(
    df: pd.DataFrame,
    detected_cols: dict,
    cfg: dict,
    model_key: str,
    prompt_key: str,
    mode_key: str,
) -> Path:
    model_cfg = cfg["models"][model_key]
    prompt_cfg = cfg["prompt_variants"][prompt_key]
    mode_cfg = cfg["context_modes"][mode_key]

    out_dir = ensure_dir(cfg.get("output_dir", "outputs/generation"))
    name = experiment_name(model_key, prompt_key, mode_key)
    out_path = out_dir / f"{name}.csv"

    answer_col = mode_cfg.get("answer_col", "answer")

    if out_path.exists():
        out_df = pd.read_csv(out_path)
        print(f"Found existing file, resuming: {out_path}")
    else:
        out_df = df.copy()
        out_df["experiment"] = name
        out_df["provider"] = model_cfg["provider"]
        out_df["generation_model"] = model_cfg["model"]
        out_df["prompt_variant"] = prompt_key
        out_df["context_mode"] = mode_key
        if answer_col not in out_df.columns:
            out_df[answer_col] = pd.NA
        out_df["generation_error"] = pd.NA

    n_rows = cfg.get("n_rows")
    if n_rows is not None:
        rows_to_process = out_df.head(int(n_rows)).index
    else:
        rows_to_process = out_df.index

    mask = is_missing(out_df.loc[rows_to_process, answer_col])
    indices = list(out_df.loc[rows_to_process].loc[mask].index)

    print(f"\n=== {name} ===")
    print(f"Rows to process: {len(indices)}")
    if not indices:
        return out_path

    provider = make_provider(model_cfg)
    max_retries = int(cfg.get("max_retries", 3))
    save_every = int(cfg.get("save_every", 1))
    sleep_between_calls = float(cfg.get("sleep_between_calls", 0.0))

    for n, idx in enumerate(tqdm(indices, desc=name), start=1):
        row = out_df.loc[idx]
        user_prompt = build_generation_prompt(row, detected_cols, mode_cfg)
        request = GenerationRequest(
            system_prompt=prompt_cfg["system_prompt"],
            user_prompt=user_prompt,
            model_cfg=model_cfg,
        )

        last_error = ""
        answer = ""
        for attempt in range(1, max_retries + 1):
            try:
                answer = provider.generate(request)
                last_error = ""
                break
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                wait = min(2**attempt, 60)
                print(f"Attempt {attempt}/{max_retries} failed for row {idx}: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        out_df.at[idx, answer_col] = answer if answer else pd.NA
        out_df.at[idx, "generation_error"] = last_error if last_error else ""

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

        if n % save_every == 0:
            out_df.to_csv(out_path, index=False)

    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG generation experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--input", type=Path, default=None, help="Optional override for input CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for output directory.")
    parser.add_argument("--n-rows", type=int, default=None, help="Optional override for number of rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input is not None:
        cfg["input_csv"] = str(args.input)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.n_rows is not None:
        cfg["n_rows"] = args.n_rows

    input_csv = Path(cfg["input_csv"])
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv.resolve()}")

    df = pd.read_csv(input_csv)
    detected_cols = detect_columns(df, cfg["columns"])
    require_columns(df, detected_cols, required=["query", "distractor", "gold_passage"])

    print(f"Loaded {len(df)} rows from {input_csv}")
    print(f"Detected columns: {detected_cols}")

    only = cfg.get("only")
    only_set = set(only) if only else None

    for model_key in cfg["models"]:
        for prompt_key in cfg["prompt_variants"]:
            for mode_key in cfg["context_modes"]:
                name = experiment_name(model_key, prompt_key, mode_key)
                if only_set is not None and name not in only_set:
                    continue
                run_one_experiment(df, detected_cols, cfg, model_key, prompt_key, mode_key)


if __name__ == "__main__":
    main()

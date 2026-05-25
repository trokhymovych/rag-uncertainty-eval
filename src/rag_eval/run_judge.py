from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI

from rag_eval.columns import detect_columns, require_columns
from rag_eval.config import load_config
from rag_eval.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt

VALID_LABELS = ["ABSTAIN", "HALLUCINATE", "HEDGE", "CORRECT"]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": VALID_LABELS},
    },
    "required": ["label"],
    "additionalProperties": False,
}


def extract_response_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    chunks = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []):
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if data.get("label") not in VALID_LABELS:
        raise ValueError(f"Invalid label: {data}")
    return data


def is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge generated RAG answers with GPT-5 structured output.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="Generated CSV to judge.")
    parser.add_argument("--output", type=Path, default=None, help="Output judged CSV. Default: input stem + __judged.csv")
    parser.add_argument("--answer-col", type=str, default=None, help="Answer column to judge. If omitted, inferred from context_mode column.")
    parser.add_argument("--model", type=str, default="gpt-5", help="OpenAI judge model.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.input)
    detected_cols = detect_columns(df, cfg["columns"])
    require_columns(df, detected_cols, required=["query", "gold_passage"])

    answer_col = args.answer_col
    if answer_col is None:
        if "context_mode" in df.columns:
            mode = str(df["context_mode"].dropna().iloc[0])
            answer_col = cfg["context_modes"].get(mode, {}).get("answer_col")
        if not answer_col:
            # Fallback for common names
            for cand in ["answer_with_distractor_and_gold", "answer_with_distractor", "answer"]:
                if cand in df.columns:
                    answer_col = cand
                    break
    if not answer_col or answer_col not in df.columns:
        raise ValueError(f"Could not infer answer column. Available columns: {list(df.columns)}")

    output = args.output or args.input.with_name(args.input.stem + "__judged.csv")
    if output.exists():
        judged_df = pd.read_csv(output)
        print(f"Found existing judged file, resuming: {output}")
    else:
        judged_df = df.copy()

    for col in ["judge_label", "judge_raw_json", "judge_error"]:
        if col not in judged_df.columns:
            judged_df[col] = pd.NA

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    indices = list(judged_df.loc[is_missing(judged_df["judge_label"])].index)
    print(f"Loaded {len(judged_df)} rows. Rows to judge: {len(indices)}")
    print(f"Answer column: {answer_col}")

    for n, idx in enumerate(tqdm(indices, desc="judging"), start=1):
        prompt = build_judge_prompt(judged_df.loc[idx], detected_cols, answer_col)
        label, raw_json, error = "", "", ""

        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.responses.create(
                    model=args.model,
                    input=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "rag_judgment",
                            "schema": JUDGE_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                raw_text = extract_response_text(response)
                parsed = parse_judge_json(raw_text)
                label = parsed["label"]
                raw_json = json.dumps(parsed, ensure_ascii=False)
                break
            except Exception as e:  # noqa: BLE001
                error = str(e)
                wait = min(2**attempt, 60)
                print(f"Attempt {attempt}/{args.max_retries} failed for row {idx}: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        judged_df.at[idx, "judge_label"] = label if label else pd.NA
        judged_df.at[idx, "judge_raw_json"] = raw_json
        judged_df.at[idx, "judge_error"] = error

        if n % args.save_every == 0:
            judged_df.to_csv(output, index=False)

    judged_df.to_csv(output, index=False)
    print(f"Saved judged CSV: {output}")
    print(judged_df["judge_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

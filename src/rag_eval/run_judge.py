from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from openai import OpenAI

load_dotenv()

from rag_eval.columns import detect_columns, require_columns
from rag_eval.config import load_config
from rag_eval.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt

VALID_LABELS = ["ABSTAIN", "INCORRECT", "CORRECT"]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": VALID_LABELS},
        "reasoning": {"type": "string"},
    },
    "required": ["label", "reasoning"],
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


_LABEL_KEYS = ("label", "classification", "category", "result", "verdict", "judgment")


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    label = next((data[k] for k in _LABEL_KEYS if data.get(k) in VALID_LABELS), None)
    if label is None:
        raise ValueError(f"Invalid label: {data}")
    return {"label": label, "reasoning": data.get("reasoning", "")}


def is_missing(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.lower() == "nan")


def _call_judge(client, provider: str, model: str, base_url: str, prompt: str, max_retries: int) -> tuple[str, str, str, str]:
    label, reasoning, raw_json, error = "", "", "", ""
    for attempt in range(1, max_retries + 1):
        try:
            if provider == "ollama":
                import urllib.request as _urllib
                payload = json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "format": JUDGE_SCHEMA,
                }).encode()
                ollama_base_url = base_url.rstrip("/").removesuffix("/v1")
                req = _urllib.Request(
                    f"{ollama_base_url}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with _urllib.urlopen(req) as resp:
                    raw_text = json.loads(resp.read()).get("message", {}).get("content", "")
            elif provider == "together":
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    chat_template_kwargs={"thinking": False},
                    stream=True,
                )
                parts = []
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        parts.append(delta)
                raw_text = "".join(parts).strip()
            else:
                response = client.responses.create(
                    model=model,
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
            reasoning = parsed.get("reasoning", "")
            raw_json = json.dumps(parsed, ensure_ascii=False)
            break
        except Exception as e:  # noqa: BLE001
            error = str(e)
            wait = min(2**attempt, 60)
            print(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return label, reasoning, raw_json, error


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge generated RAG answers with GPT-5 structured output.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="Generated CSV to judge.")
    parser.add_argument("--output", type=Path, default=None, help="Output judged CSV. Default: input stem + __judged.csv")
    parser.add_argument("--answer-col", type=str, default=None, help="Answer column to judge. If omitted, inferred from context_mode column.")
    parser.add_argument("--model", type=str, default="gpt-5", help="Judge model name.")
    parser.add_argument("--judge-provider", type=str, default="openai", choices=["openai", "ollama", "together"], help="Provider for judge model.")
    parser.add_argument("--judge-base-url", type=str, default="http://localhost:11434/v1", help="Base URL for Ollama judge.")
    parser.add_argument("--language", type=str, default=None, help="Filter rows by language column value (e.g. 'en').")
    parser.add_argument("--workers", type=int, default=4, help="Parallel API calls.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.input)
    if args.language and "language" in df.columns:
        df = df[df["language"] == args.language].reset_index(drop=True)
        print(f"Filtered to language='{args.language}': {len(df)} rows")
    detected_cols = detect_columns(df, cfg["columns"])
    require_columns(df, detected_cols, required=["query", "gold_passage"])

    answer_col = args.answer_col
    if answer_col is None:
        if "context_mode" in df.columns:
            mode = str(df["context_mode"].dropna().iloc[0])
            answer_col = cfg["context_modes"].get(mode, {}).get("answer_col")
        if not answer_col:
            # Prefer distractor+gold over distractor-only; then any answer_ column
            priority = ["answer_with_distractor_and_gold", "answer_with_distractor", "answer"]
            for cand in priority:
                if cand in df.columns:
                    answer_col = cand
                    break
            if not answer_col:
                answer_col = next(
                    (c for c in df.columns if c.startswith("answer_")), None
                )
    if not answer_col or answer_col not in df.columns:
        raise ValueError(f"Could not infer answer column. Available columns: {list(df.columns)}")

    output = args.output or args.input.with_name(args.input.stem + "__judged.csv")
    if output.exists():
        judged_df = pd.read_csv(output)
        print(f"Found existing judged file, resuming: {output}")
    else:
        judged_df = df.copy()

    for col in ["judge_label", "judge_reasoning", "judge_raw_json", "judge_error"]:
        if col not in judged_df.columns:
            judged_df[col] = pd.NA

    if args.judge_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
    elif args.judge_provider == "together":
        from together import Together
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY is not set.")
        client = Together(api_key=api_key)

    indices = list(judged_df.loc[is_missing(judged_df["judge_label"])].index)
    print(f"Loaded {len(judged_df)} rows. Rows to judge: {len(indices)}")
    print(f"Answer column: {answer_col}")
    print(f"Judge provider: {args.judge_provider}, model: {args.model}")

    print(f"\n=== Sample prompt (row {indices[0]}) ===\n{build_judge_prompt(judged_df.loc[indices[0]], detected_cols, answer_col)}\n")

    futures_to_idx = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx in indices:
            prompt = build_judge_prompt(judged_df.loc[idx], detected_cols, answer_col)
            future = executor.submit(
                _call_judge,
                client if args.judge_provider != "ollama" else None,
                args.judge_provider,
                args.model,
                args.judge_base_url,
                prompt,
                args.max_retries,
            )
            futures_to_idx[future] = idx

        for n, future in enumerate(tqdm(as_completed(futures_to_idx), total=len(indices), desc="judging"), start=1):
            idx = futures_to_idx[future]
            label, reasoning, raw_json, error = future.result()

            judged_df.at[idx, "judge_label"] = label if label else pd.NA
            judged_df.at[idx, "judge_reasoning"] = reasoning if label else pd.NA
            judged_df.at[idx, "judge_raw_json"] = raw_json
            judged_df.at[idx, "judge_error"] = error

            if n % args.save_every == 0:
                judged_df.to_csv(output, index=False)

    judged_df.to_csv(output, index=False)
    print(f"Saved judged CSV: {output}")
    print(judged_df["judge_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

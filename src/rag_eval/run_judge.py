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


def _call_judge(client, provider: str, model: str, base_url: str, prompt: str, max_retries: int, model_cfg: dict | None = None) -> tuple[str, str, str, str]:
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
            elif provider == "azure":
                cfg_a = model_cfg or {}
                kwargs_a: dict[str, Any] = {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "rag_judgment",
                            "schema": JUDGE_SCHEMA,
                            "strict": True,
                        },
                    },
                }
                if cfg_a.get("temperature") is not None:
                    kwargs_a["temperature"] = cfg_a["temperature"]
                if cfg_a.get("max_output_tokens"):
                    kwargs_a["max_completion_tokens"] = int(cfg_a["max_output_tokens"])
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs_a,
                )
                raw_text = response.choices[0].message.content or ""
            elif provider == "gemini":
                from google.genai import types
                cfg_m = model_cfg or {}
                gemini_schema = {k: v for k, v in JUDGE_SCHEMA.items() if k != "additionalProperties"}
                gen_cfg_kwargs: dict[str, Any] = {
                    "system_instruction": JUDGE_SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_schema": gemini_schema,
                }
                if cfg_m.get("temperature") is not None:
                    gen_cfg_kwargs["temperature"] = cfg_m["temperature"]
                if cfg_m.get("max_output_tokens"):
                    gen_cfg_kwargs["max_output_tokens"] = int(cfg_m["max_output_tokens"])
                thinking_budget = cfg_m.get("thinking_budget")
                if thinking_budget is not None:
                    gen_cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(thinking_budget))
                gen_cfg = types.GenerateContentConfig(**gen_cfg_kwargs)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_cfg,
                )
                raw_text = response.text.strip() if getattr(response, "text", None) else str(response)
            else:
                cfg = model_cfg or {}
                kwargs: dict[str, Any] = {}
                if cfg.get("max_output_tokens"):
                    kwargs["max_output_tokens"] = cfg["max_output_tokens"]
                if cfg.get("temperature") is not None:
                    kwargs["temperature"] = cfg["temperature"]
                reasoning_effort = cfg.get("reasoning_effort")
                if reasoning_effort is not None:
                    kwargs["reasoning"] = {"effort": str(reasoning_effort)}
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
                    **kwargs,
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
    parser.add_argument("--judge-provider", type=str, default="openai", choices=["openai", "ollama", "together", "gemini", "azure"], help="Provider for judge model.")
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
    elif args.judge_provider == "gemini":
        import google.genai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        client = genai.Client(api_key=api_key)
    elif args.judge_provider == "azure":
        from openai import AzureOpenAI
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set.")
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set.")
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    # Find matching model config entry by model name
    model_cfg = next(
        (v for v in cfg.get("models", {}).values() if v.get("model") == args.model),
        None,
    )
    print(f"\n=== Judge config ===")
    print(f"  provider : {args.judge_provider}")
    print(f"  model    : {args.model}")
    if model_cfg:
        for k, v in model_cfg.items():
            if k not in ("provider", "model", "api_key_env"):
                print(f"  {k:<16}: {v}")
    else:
        print("  (no matching model entry found in experiments.yaml — using API defaults)")
    print()

    indices = list(judged_df.loc[is_missing(judged_df["judge_label"])].index)
    print(f"Loaded {len(judged_df)} rows. Rows to judge: {len(indices)}")
    print(f"Answer column: {answer_col}")

    if not indices:
        print("All rows already judged. Skipping.")
        judged_df.to_csv(output, index=False)
        print(judged_df["judge_label"].value_counts(dropna=False))
        return

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
                model_cfg,
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

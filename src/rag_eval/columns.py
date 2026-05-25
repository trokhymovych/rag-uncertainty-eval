from __future__ import annotations

import pandas as pd


def normalize_col(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def find_col(dataframe: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by exact normalized match, then substring match."""
    normalized_to_original = {normalize_col(col): col for col in dataframe.columns}

    for cand in candidates:
        key = normalize_col(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    for cand in candidates:
        key = normalize_col(cand)
        for norm_col, original_col in normalized_to_original.items():
            if key in norm_col:
                return original_col

    return None


def detect_columns(df: pd.DataFrame, columns_cfg: dict) -> dict[str, str | None]:
    detected = {
        "query": find_col(df, columns_cfg.get("query", [])),
        "distractor": find_col(df, columns_cfg.get("distractor", [])),
        "gold_passage": find_col(df, columns_cfg.get("gold_passage", [])),
        "gold_title": find_col(df, columns_cfg.get("gold_title", [])),
    }
    return detected


def require_columns(df: pd.DataFrame, detected: dict[str, str | None], required: list[str]) -> None:
    missing = [name for name in required if detected.get(name) is None or detected[name] not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + f"\n\nDetected: {detected}"
            + f"\n\nAvailable columns: {list(df.columns)}"
        )

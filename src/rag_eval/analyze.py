from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rag_eval.columns import find_col

VALID_LABELS = ["ABSTAIN", "HALLUCINATE", "HEDGE", "CORRECT"]
VALID_DIFFICULTIES = ["easy", "medium", "hard"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze judged CSV by difficulty.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    difficulty_col = find_col(df, ["difficulty", "question_difficulty", "level", "difficulty_level"])
    if difficulty_col is None:
        raise ValueError(f"Could not find difficulty column. Available columns: {list(df.columns)}")
    if "judge_label" not in df.columns:
        raise ValueError("Input must contain judge_label column.")

    work = df.copy()
    work["difficulty_clean"] = work[difficulty_col].astype(str).str.strip().str.lower()
    work["judge_label_clean"] = work["judge_label"].astype(str).str.strip().str.upper()
    work = work[
        work["difficulty_clean"].isin(VALID_DIFFICULTIES)
        & work["judge_label_clean"].isin(VALID_LABELS)
    ].copy()

    counts = (
        work.groupby(["difficulty_clean", "judge_label_clean"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=VALID_DIFFICULTIES, fill_value=0)
        .reindex(columns=VALID_LABELS, fill_value=0)
    )
    counts["TOTAL"] = counts.sum(axis=1)
    percentages = (counts[VALID_LABELS].div(counts["TOTAL"], axis=0) * 100).round(2)
    percentages["TOTAL"] = 100.0

    summary = counts.copy()
    for label in VALID_LABELS:
        summary[f"{label.lower()}_rate"] = (summary[label] / summary["TOTAL"] * 100).round(2)

    counts.to_csv(args.output_dir / "judge_counts_by_difficulty.csv")
    percentages.to_csv(args.output_dir / "judge_percentages_by_difficulty.csv")
    summary.to_csv(args.output_dir / "judge_summary_by_difficulty.csv")

    print("Counts by difficulty:")
    print(counts)
    print("\nPercentages by difficulty:")
    print(percentages)
    print(f"\nSaved analysis files to: {args.output_dir}")


if __name__ == "__main__":
    main()

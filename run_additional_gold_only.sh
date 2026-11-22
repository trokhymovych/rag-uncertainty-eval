#!/usr/bin/env bash
# Additional experiment (camera-ready): gold-only control.
# 0) Build the input: one row per query with its gold passage (skipped if the file exists).
# 1) Generate answers with the gold passage only (2 open models × 2 prompts × 1,400 queries, Ollama).
# 2) Judge them with a single judge (gpt-5.4-mini); alone, it agrees with the expert labels as closely as the
#    three-judge majority does (93.9%, Cohen's κ = 0.86; notebooks/05_paper_results.ipynb §8).
# 3) Optional: judge a 100-row spot-check sample with the two other judges (SPOT_CHECK=1).
#
# Usage: bash run_additional_gold_only.sh
# Requires: Ollama running locally with gemma4:26b and qwen3.6:27b pulled; judge API keys in .env.
# Override defaults with env vars:
#   JUDGE_MODEL    — default: gpt-5.4-mini
#   JUDGE_PROVIDER — default: azure
#   WORKERS        — default: 10
#   SPOT_CHECK     — default: 0 (set to 1 to also run Gemini and Qwen judges on a 100-row sample)
#   QWEN_JUDGE_MODEL / QWEN_JUDGE_PROVIDER — required when SPOT_CHECK=1 (same setup as the original Qwen3.7 Plus judge)

set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-azure}"
WORKERS="${WORKERS:-10}"
SPOT_CHECK="${SPOT_CHECK:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="configs/additional_gold_only.yaml"
EVAL_CSV="data/rag_eval_multilingual_eval.csv"
INPUT_CSV="data/additional/gold_only_input.csv"
GEN_DIR="data/additional/generation"
JUDGE_DIR="data/additional/judge"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# ── 0. Gold-only input ────────────────────────────────────────────────────────
# One row per query (1,400) with the gold passage used in the distractor_plus_gold run.
# The gold passage is identical across the three tiers of a query, so tiers are collapsed.
if [[ ! -f "$INPUT_CSV" ]]; then
  mkdir -p "$(dirname "$INPUT_CSV")"
  "$PYTHON" - "$EVAL_CSV" "$INPUT_CSV" <<'PY'
import sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
ev = pd.read_csv(src)
cols = ['language', 'query_id', 'query', 'gold_generation_id', 'gold_generation_text', 'gold_judge_text', 'n_positives']
gold = ev[cols].drop_duplicates(['language', 'query_id'])
assert len(gold) == ev.groupby(['language', 'query_id']).ngroups
assert (ev.groupby(['language', 'query_id'])['gold_generation_id'].nunique() == 1).all()
# run_generation.py requires a distractor column; it is never used (include_distractor: false).
gold.insert(gold.columns.get_loc('gold_generation_id'), 'distractor_text', '')
gold.to_csv(out, index=False)
print(f'Wrote {out}: {len(gold):,} queries')
PY
fi

# ── 1. Generation ─────────────────────────────────────────────────────────────
"$PYTHON" src/rag_eval/run_generation.py --config "$CONFIG"

# ── 2. Single-judge labeling ──────────────────────────────────────────────────
mkdir -p "$JUDGE_DIR/openai"
for f in "$GEN_DIR"/*__gold_only.csv; do
  name="$(basename "$f" .csv)"
  "$PYTHON" src/rag_eval/run_judge.py \
    --config "$CONFIG" \
    --input "$f" \
    --output "$JUDGE_DIR/openai/${name}__judged.csv" \
    --model "$JUDGE_MODEL" \
    --judge-provider "$JUDGE_PROVIDER" \
    --workers "$WORKERS"
done

# ── 3. Optional spot check with the two other judges ─────────────────────────
if [[ "$SPOT_CHECK" == "1" ]]; then
  SAMPLE="$GEN_DIR/spot_check_sample.csv"
  "$PYTHON" - <<'EOF'
import glob
import pandas as pd
frames = [pd.read_csv(f) for f in sorted(glob.glob('data/additional/generation/*__gold_only.csv'))]
pd.concat(frames, ignore_index=True).sample(100, random_state=42).to_csv(
    'data/additional/generation/spot_check_sample.csv', index=False)
EOF
  mkdir -p "$JUDGE_DIR/gemini" "$JUDGE_DIR/qwen"
  "$PYTHON" src/rag_eval/run_judge.py --config "$CONFIG" --input "$SAMPLE" \
    --output "$JUDGE_DIR/gemini/spot_check_sample__judged.csv" \
    --model "${GEMINI_JUDGE_MODEL:-gemini-3.1-flash-lite}" --judge-provider gemini --workers "$WORKERS"
  "$PYTHON" src/rag_eval/run_judge.py --config "$CONFIG" --input "$SAMPLE" \
    --output "$JUDGE_DIR/qwen/spot_check_sample__judged.csv" \
    --model "${QWEN_JUDGE_MODEL:?set QWEN_JUDGE_MODEL}" --judge-provider "${QWEN_JUDGE_PROVIDER:?set QWEN_JUDGE_PROVIDER}" --workers "$WORKERS"
fi

#!/usr/bin/env bash
# Judge all generation CSVs with OpenAI.
# Usage: bash run_judge_all.sh
# Override defaults with env vars:
#   JUDGE_MODEL    — default: gemini-3.1-flash-lite
#   JUDGE_PROVIDER — default: openai
#   WORKERS        — default: 4 (parallel API calls per file)

set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.1-flash-lite}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-gemini}"
WORKERS="${WORKERS:-4}"
PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="configs/experiments.yaml"
OUTPUT_DIR="data/judge step/gemini"

mkdir -p "$OUTPUT_DIR"

# ── Files to judge ────────────────────────────────────────────────────────────
# Add or remove paths/globs as needed.
INPUT_FILES=(
  # Multilingual API model files (data/generation step/)
  "data/generation step/multilingual_claude_sonnet4_answers.csv"
  "data/generation step/multilingual_claude_sonnet4_answers_advanced.csv"
  "data/generation step/multilingual_gemini31_flash_lite_answers.csv"
  "data/generation step/multilingual_gemini31_flash_lite_answers_advanced.csv"
  "data/generation step/multilingual_gemma4_answers.csv"
  "data/generation step/multilingual_gemma4_answers_advanced.csv"
  "data/generation step/multilingual_gpt54mini_answers.csv"
  "data/generation step/multilingual_gpt54mini_answers_advanced.csv"
  "data/generation step/multilingual_qwen36_answers.csv"
  "data/generation step/multilingual_qwen36_answers_advanced.csv"

  # Multilingual open-source model files (data/generation step/withgold)
  "data/generation step/withgold/gemma4_26b__advanced__distractor_plus_gold.csv"
  "data/generation step/withgold/gemma4_26b__simple__distractor_plus_gold.csv"
  "data/generation step/withgold/qwen3.6:27b__advanced__distractor_plus_gold.csv"
  "data/generation step/withgold/qwen3.6:27b__simple__distractor_plus_gold.csv"
)

TOTAL=${#INPUT_FILES[@]}
DONE=0
FAILED=()

for INPUT in "${INPUT_FILES[@]}"; do
  if [[ ! -f "$INPUT" ]]; then
    echo "⚠️  File not found, skipping: $INPUT"
    continue
  fi

  # Output to dedicated judge directory, preserving the input filename
  BASENAME="$(basename "${INPUT%.*}")"
  OUTPUT="${OUTPUT_DIR}/${BASENAME}__judged.csv"

  echo "────────────────────────────────────────────────"
  echo "[$((DONE+1))/$TOTAL] $INPUT"
  echo "  → $OUTPUT"

  PYTHONPATH=src "$PYTHON" -m rag_eval.run_judge \
    --config "$CONFIG" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --judge-provider "$JUDGE_PROVIDER" \
    --model "$JUDGE_MODEL" \
    --workers "$WORKERS" \
    --save-every 10 \
    && DONE=$((DONE+1)) \
    || { echo "❌  FAILED: $INPUT"; FAILED+=("$INPUT"); }
done

echo "════════════════════════════════════════════════"
echo "Done: $DONE / $TOTAL"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed files:"
  for f in "${FAILED[@]}"; do echo "  $f"; done
  exit 1
fi

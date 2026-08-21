# Does RAG Answer Without Sufficient Evidence? Reusing IR Benchmarks for RAG Uncertainty Evaluation

Code for the paper **"Does RAG Answer Without Sufficient Evidence? Reusing IR Benchmarks for RAG Uncertainty Evaluation"**.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Overview

This repository implements a methodology for converting IR hard-negative datasets into difficulty-stratified RAG uncertainty benchmarks. We use hard negatives from **MIRACL Hard Negatives** across 14 languages, graded by similarity to the gold passage, and evaluate whether LLMs abstain or answer incorrectly when given a topically relevant but answer-free context.

Each benchmark example contains a question, a hard-negative distractor passage (used as retrieved context), and a gold reference passage (used for evaluation only). Model responses are classified as `ABSTAIN`, `CORRECT`, or `INCORRECT` using an ensemble of three LLM judges validated against expert human annotation.

---

## Repository Structure

```text
rag-uncertainty-eval/
├── configs/                        # Experiment configuration files (system prompts inside)
│   ├── experiments.yaml            # Frontier models (GPT, Gemini, Claude)
│   ├── experiments_en_ollama.yaml  # Ollama inference (English)
│   ├── experiments_gemma_all.yaml  # Gemma 4 26B (all languages)
│   └── experiments_qwen_all.yaml   # Qwen3.6 27B (all languages)
├── data/
│   ├── rag_eval_multilingual_eval.csv  # Main multilingual benchmark (14 languages)
│   ├── manual_labeling_samples.csv     # Subset prepared for human annotation
│   ├── expert_annotation/              # Expert annotator labels
│   ├── generation step/                # Raw generation outputs
│   └── judge step/                     # Judged generation outputs
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Benchmark construction from MIRACL
│   ├── 02_check_generations.ipynb      # Sanity checks on generated answers
│   ├── 03_prepare_manual_labeling.ipynb
│   ├── 04_annotator_agreement.ipynb    # Inter-annotator agreement analysis
│   └── 05_paper_figures.ipynb          # All figures and tables in the paper
├── outputs/                        # Generated outputs, analysis tables, figures
├── resources/
│   └── annotation_guidlines.md     # Human annotation guidelines (label definitions, priority order)
├── src/
│   └── rag_eval/
│       ├── prompts.py              # Prompt construction for generation and judging
│       ├── clients.py              # API clients: OpenAI, Anthropic, Gemini, Ollama, Together
│       ├── run_generation.py       # Answer generation script
│       ├── run_judge.py            # LLM-as-a-judge classification script
│       ├── analyze.py              # Aggregation and metrics by difficulty tier
│       ├── config.py               # Config loading utilities
│       └── columns.py              # CSV column detection utilities
├── run_judge_all.sh                # Batch judging with OpenAI
├── run_judge_all_gemini.sh         # Batch judging with Gemini
├── requirements.txt
└── LICENSE
```

---

## Setup

```bash
git clone https://github.com/anonymous/rag-uncertainty-eval.git
cd rag-uncertainty-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Set API keys (or place them in a `.env` file):

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

For open-weight models (Qwen, Gemma), install and run [Ollama](https://ollama.com/) locally and pull the required models.

---

## Configuration

Each YAML file in `configs/` defines a full experiment: input data, models, prompt variants, context modes, and output settings. **System prompts are stored directly inside the config files** under `prompt_variants.<name>.system_prompt`.

The two prompt variants used in the paper:

- **Simple** — instructs the model to use only the provided context and keep the answer concise.
- **Advanced** — adds an explicit abstention instruction: *"If the context does not contain enough information to answer the question, say: 'I don't know based on the provided context.'"*

The two context modes:

- **distractor** — model receives only the hard-negative distractor as context.
- **distractor_plus_gold** — model receives both the distractor and the correct passage (robustness experiment).

See [`src/rag_eval/prompts.py`](src/rag_eval/prompts.py) for prompt assembly logic and the judge prompt.

---

## Running the Pipeline

### 1. Generate answers

```bash
python src/rag_eval/run_generation.py --config configs/experiments.yaml
```

Outputs one CSV per experiment combination (`<model>__<prompt>__<context_mode>.csv`) in the directory set by `output_dir` in the config. The script is resumable — it skips rows that already have answers.

### 2. Judge answers

Classify answers as `ABSTAIN`, `CORRECT`, or `INCORRECT`:

```bash
python src/rag_eval/run_judge.py \
  --config configs/experiments.yaml \
  --input outputs/generation/gpt54mini__simple__distractor.csv
```

Output is saved with `__judged` appended to the filename. To judge all files at once:

```bash
bash run_judge_all.sh        # uses OpenAI judge
bash run_judge_all_gemini.sh # uses Gemini judge
```

The paper uses a majority vote of three judges: GPT-5.4-mini, Gemini-3.1-flash-lite, and Qwen3.7 Plus.

### 3. Analyze results

```bash
python src/rag_eval/analyze.py \
  --input outputs/judged/gpt54mini__simple__distractor__judged.csv \
  --output-dir outputs/tables/gpt54mini__simple__distractor
```

Produces `judge_counts_by_difficulty.csv` and `judge_percentages_by_difficulty.csv`.

All figures and tables from the paper are reproduced in [`notebooks/05_paper_figures.ipynb`](notebooks/05_paper_figures.ipynb).

---

## Human Annotation Guidelines

Annotation guidelines used by expert annotators for judge validation are in [`resources/annotation_guidlines.md`](resources/annotation_guidlines.md). Each response is assigned exactly one label (`ABSTAIN`, `CORRECT`, `INCORRECT`) based on the gold reference only. Labels are applied in priority order: ABSTAIN > CORRECT > INCORRECT.

---

## Citation

```bibtex
TBD
```

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

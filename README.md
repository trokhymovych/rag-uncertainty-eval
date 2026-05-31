# rag-uncertainty-eval

Code for **"Does the RAG Know It Doesn't Know? IR Benchmarks for RAG Uncertainty Evaluation"**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Overview

We evaluate whether RAG models know when they can't answer. Given a topically relevant but **answer-free** passage, does the model hallucinate, abstain, or hedge? We measure how well stated confidence matches actual accuracy.

The model answers using the retrieved context, and its response is classified into one of four labels:

- `ABSTAIN`
- `HALLUCINATE`
- `HEDGE`
- `CORRECT`

Models tested: **GPT-5.5, Claude Sonnet 4.6, Gemma 4 31B, Gemma 4 E2B, Qwen3.6-27B**  
Dataset: **MIRACL English** — 300 human-annotated negative passages

---

## Dataset

The benchmark is built from the English split of **MIRACL Hard Negatives**.

Distractor passages are stratified into three difficulty levels using TF-IDF cosine similarity between the distractor passage and the gold passage:

- `easy`
- `medium`
- `hard`

The final benchmark contains 300 examples:

```text
100 easy distractors
100 medium distractors
100 hard distractors
```

---

## Experiments

The pipeline evaluates four generation models:

```text
GPT-5
GPT-4.1
Claude Sonnet 4
Gemini 3.1 Flash-Lite
```

Each model is tested with two prompt variants:

```text
simple
advanced
```

and two context modes:

```text
distractor
distractor_plus_gold
```

This gives:

```text
4 models × 2 prompts × 2 context modes = 16 generation experiments
```

Generated answers are classified using GPT-5 as an LLM-as-a-judge.

---

## Setup

Clone the repository:

```
git clone https://github.com/<your-org>/rag-uncertainty-eval.git
cd rag-uncertainty-eval
```

Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```
Add the project source path:
```
export PYTHONPATH=$PWD/src:$PYTHONPATH
```
---

## Models

The experiment pipeline supports four generation models:

```text
GPT-5
GPT-4.1
Claude Sonnet 4
Gemini 3.1 Flash-Lite
```

Each model is evaluated under two prompt variants:

- Simple
- Advanced

and two context modes:

- `distractor`
- `distractor_plus_gold`

This gives:

```text
4 models × 2 prompts × 2 context modes = 16 generation experiments
```

Generated answers are then classified with GPT-5 as an LLM-as-judge.

---

## Usage

```bash
# 1. Sample distractors from MIRACL
python src/dataset/load_miracl.py --n 300 --out data/processed/distractors_300.jsonl

# 2. Run models
python src/rag/run_models.py --input data/processed/distractors_300.jsonl \
  --models gpt-5.5 claude-sonnet-4-6 gemma-4-31b gemma-4-e2b qwen3.6-27b

# 3. Classify responses (LLM-as-judge)
python src/judge/classify_responses.py --input results/raw_outputs/ --judge gpt-5.5

# 4. Compute metrics
python src/metrics/calibration.py --input results/tables/classifications.csv
```

---

## Repository Structure

```
rag-uncertainty-eval/
├── configs/
│   └── experiments.yaml          # experiment configuration
├── src/
│   └── rag_eval/
│       ├── config.py             # config loading utilities
│       ├── columns.py            # column detection helpers
│       ├── prompts.py            # simple and advanced prompts
│       ├── clients.py            # OpenAI, Anthropic, Gemini clients
│       ├── run_generation.py     # generation pipeline
│       ├── run_judge.py          # LLM-as-judge classification
│       └── analyze.py            # result analysis
├── notebooks/
│   ├── 01-miracl-hard-negatives-eda.ipynb
│   ├── 02-eval-dataset-construction.ipynb
│   ├── 03-gpt-distractor-answering.ipynb
│   └── 04-gpt5-llm-as-judge.ipynb
├── paper/                        # paper source files
├── results/                      # generated results, usually gitignored
├── README.md
├── requirements.txt
└── LICENSE
---
```

## Citation

```bibtex
TBA
```
---
## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

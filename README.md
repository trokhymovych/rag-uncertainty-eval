# rag-uncertainty-eval

Code for **"Does the Agent Know It Doesn't Know? Evaluating Uncertainty Awareness in RAG Systems"**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What this is

We evaluate whether RAG models know when they can't answer. Given a topically relevant but **answer-free** passage, does the model hallucinate, abstain, or hedge? We measure how well stated confidence matches actual accuracy.

Models tested: **GPT-5.5, Claude Sonnet 4.6, Gemma 4 31B, Gemma 4 E2B, Qwen3.6-27B**  
Dataset: **MIRACL English** — 300 human-annotated negative passages

---

## Setup

```bash
git clone https://github.com/<your-org>/rag-uncertainty-eval.git
cd rag-uncertainty-eval
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

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

## Project structure

```
src/
  dataset/      # MIRACL loader
  rag/          # prompt templates + model API loop
  judge/        # LLM-as-judge classifier + human validation (Cohen's kappa)
  metrics/      # ECE, Brier score, overconfidence gap
notebooks/      # exploration and analysis
results/        # outputs (gitignored)
configs/        # model names and API versions
```

---

## Citation

```bibtex
TBA
```
---
## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
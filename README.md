# rag-uncertainty-eval

Code for **“Does the RAG Know It Doesn't Know? IR Benchmarks for RAG Uncertainty Evaluation”**.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## Overview

This repository contains code for evaluating uncertainty awareness in Retrieval-Augmented Generation (RAG) systems.

The main research question is:

> When a model receives a topically relevant but answer-free retrieved passage, does it abstain, hallucinate, hedge, or answer correctly?

We convert hard negatives from an information retrieval benchmark into a controlled RAG uncertainty benchmark. Each example contains:

- a question,
- a gold passage used only for evaluation,
- a hard-negative distractor passage used as retrieved context.

The model answers using the retrieved context, and its response is classified into one of four labels:

- `ABSTAIN`
- `HALLUCINATE`
- `HEDGE`
- `CORRECT`

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

## Repository Structure

```text
rag-uncertainty-eval/
├── configs/
│   └── experiments.yaml
├── notebooks/
│   ├── 01-miracl-hard-negatives-eda.ipynb
│   └── 02-eval-dataset-construction.ipynb
├── results/
├── src/
│   └── rag_eval/
│       ├── __init__.py
│       ├── analyze.py
│       ├── clients.py
│       ├── columns.py
│       ├── config.py
│       ├── prompts.py
│       ├── run_generation.py
│       └── run_judge.py
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/rag-uncertainty-eval.git
cd rag-uncertainty-eval
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add the project source path:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Set API keys:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

---

## Configuration

Main experiment settings are stored in:

```text
configs/experiments.yaml
```

The configuration file controls:

- input CSV path,
- output directory,
- generation models,
- prompt variants,
- context modes,
- column aliases,
- retry behavior,
- saving behavior.

The expected input file is:

```text
data/rag_eval_en_300.csv
```

The input CSV should contain columns for:

- question or query,
- distractor passage,
- gold passage,
- optional gold passage title,
- difficulty label.

---

## Running Generation

Run all configured generation experiments:

```bash
python src/rag_eval/run_generation.py \
  --config configs/experiments.yaml
```

Run generation with a custom input file:

```bash
python src/rag_eval/run_generation.py \
  --config configs/experiments.yaml \
  --input data/rag_eval_en_300.csv
```

Run only a small test sample:

```bash
python src/rag_eval/run_generation.py \
  --config configs/experiments.yaml \
  --n-rows 5
```

Save outputs to a custom directory:

```bash
python src/rag_eval/run_generation.py \
  --config configs/experiments.yaml \
  --output-dir outputs/generation
```

Each experiment is saved as a separate CSV file:

```text
<model>__<prompt>__<context_mode>.csv
```

Example:

```text
gpt5__simple__distractor.csv
gpt5__advanced__distractor_plus_gold.csv
```

The generation script supports resuming. If an output file already exists, it continues from missing answers instead of rerunning completed rows.

---

## Running the LLM Judge

After generation, classify model answers with the judge:

```bash
python src/rag_eval/run_judge.py \
  --config configs/experiments.yaml \
  --input outputs/generation/gpt5__simple__distractor.csv
```

The judged file is saved with `__judged` added to the filename:

```text
gpt5__simple__distractor__judged.csv
```

You can also specify the output path manually:

```bash
python src/rag_eval/run_judge.py \
  --config configs/experiments.yaml \
  --input outputs/generation/gpt5__simple__distractor.csv \
  --output outputs/judged/gpt5__simple__distractor__judged.csv
```

The judge assigns one of four labels:

```text
ABSTAIN
HALLUCINATE
HEDGE
CORRECT
```

---

## Analyzing Results

Compute counts and percentages by difficulty level:

```bash
python src/rag_eval/analyze.py \
  --input outputs/generation/gpt5__simple__distractor__judged.csv \
  --output-dir outputs/analysis/gpt5__simple__distractor
```

This creates:

```text
judge_counts_by_difficulty.csv
judge_percentages_by_difficulty.csv
judge_summary_by_difficulty.csv
```

---

## Prompt Variants

### Simple prompt

The simple prompt instructs the model to answer using only the provided context and keep the answer concise.

### Advanced prompt

The advanced prompt adds an explicit abstention instruction:

```text
If the context does not contain enough information to answer the question, say:
"I don't know based on the provided context."
```

This allows us to test whether explicit abstention instructions reduce hallucination under misleading retrieval.

---

## Context Modes

### Distractor-only

The model receives only the hard-negative distractor passage as context.

This setting tests whether the model can recognize that the retrieved passage is insufficient.

### Distractor + gold

The model receives both the distractor passage and the gold passage.

This setting tests whether the model can recover the correct answer when relevant evidence is mixed with misleading retrieved context.

---


## Citation

TBD

---

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
from __future__ import annotations

import pandas as pd


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def build_context(row: pd.Series, detected_cols: dict, mode_cfg: dict) -> str:
    """Build context without adding titles or labels inside the context text."""
    parts: list[str] = []

    if mode_cfg.get("include_distractor", False):
        distractor_col = detected_cols.get("distractor")
        if distractor_col:
            text = clean_text(row.get(distractor_col, "")).strip()
            if text:
                parts.append(text)

    if mode_cfg.get("include_gold", False):
        gold_col = detected_cols.get("gold_passage")
        if gold_col:
            text = clean_text(row.get(gold_col, "")).strip()
            if text:
                parts.append(text)

    return "\n\n".join(parts)


def build_generation_prompt(row: pd.Series, detected_cols: dict, mode_cfg: dict) -> str:
    query_col = detected_cols["query"]
    question = clean_text(row.get(query_col, ""))
    context = build_context(row, detected_cols, mode_cfg)

    return f"""Question:
{question}

Context passage:
{context}

Answer:"""


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of RAG answers.

Your task is to classify a MODEL RESPONSE into exactly one category:
ABSTAIN, CORRECT, or INCORRECT.

Use only the GOLD PASSAGE text as the source of truth.
Do not use outside knowledge.
Return only valid JSON matching the provided schema."""


def build_judge_prompt(row: pd.Series, detected_cols: dict, answer_col: str) -> str:
    question = clean_text(row.get(detected_cols["query"], ""))
    gold_passage = clean_text(row.get(detected_cols.get("gold_judge_passage") or detected_cols["gold_passage"], ""))
    model_output = clean_text(row.get(answer_col, ""))

    return f"""SETUP: The model was given a context passage to answer the question. That context passage
may differ from the GOLD PASSAGE below. The GOLD PASSAGE is used only to verify whether
the model's answer is factually correct — it was NOT shown to the model.
Therefore: if the model says it cannot answer from its context, label ABSTAIN regardless
of whether the GOLD PASSAGE contains the answer.

Classify the following MODEL RESPONSE. Apply the definitions below in priority order:

1. ABSTAIN
   The MODEL RESPONSE says that the provided context does not contain
   enough information to answer the question. Apply ABSTAIN even when the MODEL RESPONSE
   also describes what the context does contain
   (e.g. "the text does not state X; it only mentions Y" → ABSTAIN).
   Also apply ABSTAIN when the MODEL RESPONSE provides only a partial answer and explicitly
   states the context is insufficient for a complete answer.
   A response that abstains is NEVER labeled CORRECT, even if the GOLD PASSAGE
   also does not answer the question.

2. CORRECT
   The MODEL RESPONSE gives a direct factual answer to the question whose substance is
   directly supported by the GOLD PASSAGE. Exact wording is not required, but the claim
   must be traceable to the GOLD PASSAGE text.
   CORRECT applies only to direct answers to the question — never to statements about
   what the context contains or does not contain. Those are always ABSTAIN.

3. INCORRECT  (only when neither of the above fit)
   The MODEL RESPONSE presents an answer whose substance is not supported by or contradicts
   the GOLD PASSAGE. This includes answers that are not supported by the GOLD PASSAGE.

Decision order:
- Is the MODEL RESPONSE primarily a statement about what the context contains or lacks? → ABSTAIN
- Is the MODEL RESPONSE a direct factual answer to the question, supported by the GOLD PASSAGE? → CORRECT
- Otherwise → INCORRECT

# QUESTION:
{question}

# GOLD PASSAGE:
{gold_passage}

# MODEL RESPONSE:
{model_output}

Return JSON with exactly two fields:
- "label": one of ABSTAIN, CORRECT, INCORRECT
- "reasoning": one sentence of up to 10 words explaining the label"""

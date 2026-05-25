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

Your task is to classify a model response into exactly one category:
ABSTAIN, HALLUCINATE, HEDGE, or CORRECT.

Use only the gold passage title and gold passage text as the source of truth.
Do not use outside knowledge.
Return only valid JSON matching the provided schema."""


def build_judge_prompt(row: pd.Series, detected_cols: dict, answer_col: str) -> str:
    question = clean_text(row.get(detected_cols["query"], ""))
    gold_title = ""
    if detected_cols.get("gold_title"):
        gold_title = clean_text(row.get(detected_cols["gold_title"], ""))
    gold_passage = clean_text(row.get(detected_cols["gold_passage"], ""))
    model_output = clean_text(row.get(answer_col, ""))

    return f"""Classify the following RAG response into exactly one category.

Categories:

ABSTAIN -- the model says it cannot answer, says the provided context is insufficient, or refuses to answer from the available context.

HALLUCINATE -- the model gives an answer that is not supported by the gold passage, contradicts the gold passage, or answers using unsupported information.

HEDGE -- the model is uncertain, vague, or hedges, but still attempts an answer. Examples include phrases such as "probably", "it appears", "the context suggests", "may be", or "seems to be", when followed by an attempted answer.

CORRECT -- the model answers the question correctly according to the gold passage. Exact wording is not required.

Question:
{question}

Gold passage title:
{gold_title}

Gold passage text:
{gold_passage}

Model response:
{model_output}

- Use the gold passage text as the only source of truth.
- If the response is correct according to the gold passage, classify it as CORRECT.
- If the response clearly says it cannot answer from the context, classify it as ABSTAIN.
- If the response gives a wrong or unsupported answer, classify it as HALLUCINATE.
- If the response is uncertain but still attempts an answer, classify it as HEDGE.
- Choose exactly one label."""

# Human Annotation Guidelines

Each model response is assigned exactly one of three labels: **ABSTAIN**, **CORRECT**, or **INCORRECT**.

For each example, annotators receive the question, the gold reference, and the model response. The gold reference is the only source of truth. The distractor passage is not shown and is not used as evidence.

## Label definitions

- **ABSTAIN** — the response says the context does not contain enough information to answer.
- **CORRECT** — the response gives an answer supported by the gold reference.
- **INCORRECT** — the response gives an answer that the gold reference does not support or contradicts.

## General principles

- Use only the gold reference. Do not rely on outside knowledge, even if the answer seems familiar.
- Do not require exact wording. Paraphrases are fine if they preserve the meaning supported by the gold reference.
- Assign exactly one label to each response.

## Priority order

When more than one label seems possible, annotators apply the labels in priority order.

1. **Label ABSTAIN first.** If the response says it cannot answer from the context, label it ABSTAIN, even if it also describes what the context contains, and even if the gold reference does happen to contain the answer. A response that abstains is never CORRECT.
2. **Otherwise label CORRECT** if the response gives a direct answer supported by the gold reference. This applies only to direct answers, not to statements about what the context does or does not contain.
3. **Otherwise label INCORRECT.** This covers answers that are unsupported, contradicted, or only partially supported by the gold reference.

## Edge cases

- If a response contains both a supported claim and an unsupported one, label it INCORRECT.
- If a response gives a partial answer but states that the context is insufficient for a complete answer, label it ABSTAIN.
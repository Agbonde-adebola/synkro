"""
Example: BERT Toxicity Detection Dataset

Generate a dataset for training a BERT classifier to detect toxic content.
"""

import synkro
from synkro.examples import TOXICITY_POLICY
from synkro.formatters import BERTFormatter
from synkro.models.google import Google


def main():
    result = synkro.generate(
        TOXICITY_POLICY,
        traces=50,
        generation_model=Google.GEMINI_25_FLASH,
        grading_model=Google.GEMINI_25_FLASH,
    )

    formatter = BERTFormatter(
        task="classification",
        text_field="user",
        label_field="scenario_type",
        label_map={
            "positive": 0,
            "negative": 1,
            "edge_case": 1,
        },
    )

    formatter.save(result.dataset.traces, "toxicity_bert.jsonl")


if __name__ == "__main__":
    main()

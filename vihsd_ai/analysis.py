"""Error aggregates and predictions for synthetic examples; no plots or raw exports."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import fingerprint


@dataclass
class ErrorAnalysis:
    model_name: str
    counts: pd.DataFrame
    rates: pd.DataFrame
    by_length: list
    examples: list
    payload: dict


def analyze_errors(test):
    test_df, test_prediction, test_metrics = test.frame, test.prediction, test.metrics
    lock = test.lock
    analysis = pd.DataFrame(
        {"text": test_df.text, "truth": test_df.label, "prediction": test_prediction}
    )
    analysis["correct"] = analysis.truth == analysis.prediction
    analysis["length"] = analysis.text.str.len()
    mistakes = analysis.loc[~analysis.correct]
    cm = np.asarray(test_metrics["confusion_matrix"]["rows_true_columns_predicted"])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    assert len(mistakes) == fp + fn
    error_counts = pd.DataFrame(
        [
            {"outcome": "TN: SAFE → SAFE", "count": tn},
            {"outcome": "FP: SAFE → TOXIC", "count": fp},
            {"outcome": "FN: TOXIC → SAFE", "count": fn},
            {"outcome": "TP: TOXIC → TOXIC", "count": tp},
        ]
    )
    error_counts["fraction_of_test"] = error_counts["count"] / len(test_df)
    error_rates = pd.DataFrame(
        [
            {
                "rate": "False positive rate: FP / số SAFE thật",
                "value": fp / (tn + fp) if tn + fp else np.nan,
                "denominator": tn + fp,
            },
            {
                "rate": "False negative rate: FN / số TOXIC thật",
                "value": fn / (tp + fn) if tp + fn else np.nan,
                "denominator": tp + fn,
            },
            {
                "rate": "Error rate: (FP+FN) / toàn bộ test",
                "value": (fp + fn) / len(test_df),
                "denominator": len(test_df),
            },
        ]
    )
    length_labels = ["0–50", "51–100", "101–200", "201–500", ">500"]
    analysis["length_group"] = pd.cut(
        analysis.length, bins=[-1, 50, 100, 200, 500, np.inf], labels=length_labels
    )
    length_rows = []
    for group in length_labels:
        subset = analysis.loc[analysis.length_group == group]
        count = len(subset)
        errors = int((~subset.correct).sum())
        length_rows.append(
            {
                "length_chars": group,
                "samples": count,
                "errors": errors,
                "error_rate": errors / count if count else None,
            }
        )
    error_examples = [
        {
            "fingerprint_prefix": fingerprint(row.text)[:16],
            "truth": row.truth,
            "prediction": row.prediction,
            "length_chars": int(row.length),
        }
        for row in mistakes.head(20).itertuples()
    ]
    private_errors = {
        "model": lock["selected_model"],
        "privacy": "aggregate_and_fingerprint_only",
        "post_hoc": True,
        "total_errors": len(mistakes),
        "SAFE_as_TOXIC": fp,
        "TOXIC_as_SAFE": fn,
        "by_length": length_rows,
        "examples": error_examples,
        "fingerprint_prefixes": [row["fingerprint_prefix"] for row in error_examples],
    }

    return ErrorAnalysis(
        test.model_name,
        error_counts,
        error_rates,
        length_rows,
        error_examples,
        private_errors,
    )


def system_decision(model, model_name, text):
    label = str(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        confidence = float(model.predict_proba([text])[0][classes.index(label)])
    else:
        score = float(model.decision_function([text])[0])
        p_toxic = 1 / (1 + math.exp(-max(-50, min(50, score))))
        confidence = p_toxic if label == "TOXIC" else 1 - p_toxic
    uncertain = confidence < 0.65
    return {
        "model": model_name,
        "label": label,
        "confidence_proxy": round(confidence, 6),
        "warning_level": ("HIGH" if confidence >= 0.8 else "MEDIUM")
        if label == "TOXIC"
        else ("REVIEW" if uncertain else "LOW"),
        "warning": "Chuyển human review"
        if label == "TOXIC"
        else "Cho phép báo cáo thủ công",
        "human_review": label == "TOXIC" or uncertain,
        "flow": ["input", "preprocess", "SAFE_or_TOXIC", "warning", "human_review"],
        "privacy": "input_not_echoed",
        "scope": "toxicity_not_fake_news",
    }


def predict_examples(model, model_name, sentences):
    rows = []
    for index, sentence in enumerate(sentences, 1):
        result = system_decision(model, model_name, sentence)
        rows.append(
            {
                "example": f"DEMO-{index:02d}",
                "synthetic_text": sentence,
                "model": model_name,
                "prediction": result["label"],
                "confidence_proxy": result["confidence_proxy"],
                "warning_level": result["warning_level"],
                "human_review": result["human_review"],
                "warning": result["warning"],
            }
        )
    return rows

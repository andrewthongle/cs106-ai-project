"""Binary evaluation and the unchanged dev selection rule."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .config import BINARY_LABELS


def metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=BINARY_LABELS, zero_division=0
    )
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=BINARY_LABELS, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=BINARY_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted": {"precision": float(wp), "recall": float(wr), "f1": float(wf)},
        "macro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "per_class": {
            label: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, label in enumerate(BINARY_LABELS)
        },
        "confusion_matrix": {
            "labels": BINARY_LABELS,
            "rows_true_columns_predicted": cm.tolist(),
        },
    }


def selection_key(score):
    return score["macro"]["f1"], score["per_class"]["TOXIC"]["recall"]


def rank_models(rows):
    return sorted(
        rows,
        key=lambda row: (
            -row["dev"]["macro"]["f1"],
            -row["dev"]["per_class"]["TOXIC"]["recall"],
            row["name"],
        ),
    )

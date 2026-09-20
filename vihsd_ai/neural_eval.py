"""Test-set evaluation for the neural branch, opened only after the dev lock.

Mirrors the baseline protocol in baselines.evaluate_test: dev picks the epoch and
the decision threshold, the lock file is verified, and only then is test read.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import write_json
from .config import BINARY_LABELS
from .data import load_vihsd_split, stratified_sample
from .metrics import metrics
from .neural import encode_texts, make_bilstm
from .preprocessing import SocialPreprocessor

DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclass
class NeuralTestResults:
    models: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    frame: Any = None
    payload: dict = field(default_factory=dict)


def toxic_probabilities(logits):
    """Xác suất lớp TOXIC bằng softmax; BINARY_LABELS = [SAFE, TOXIC] nên TOXIC là cột 1."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return (exponent / exponent.sum(axis=1, keepdims=True))[:, 1]


def labels_from_probabilities(probabilities, threshold=0.5):
    return [
        BINARY_LABELS[1] if p >= threshold else BINARY_LABELS[0] for p in probabilities
    ]


def bilstm_logits(checkpoint_path, processed_texts, device, *, batch_size=64):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture") or {}
    vocabulary = checkpoint["vocabulary"]
    model = make_bilstm(
        architecture.get("vocabulary_size", len(vocabulary)),
        architecture.get("embedding_dim", 128),
        architecture.get("hidden_size", 96),
        architecture.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x, lengths = encode_texts(
        processed_texts, vocabulary, architecture.get("max_length", 128)
    )
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch_logits = model(
                x[start : start + batch_size].to(device), lengths[start : start + batch_size]
            )
            outputs.append(batch_logits.float().cpu().numpy())
    return np.concatenate(outputs), checkpoint.get("best_epoch")


def phobert_logits(model_dir, processed_texts, device, *, batch_size=32, max_length=128):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(processed_texts), batch_size):
            encoded = tokenizer(
                processed_texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs.append(model(**encoded).logits.float().cpu().numpy())
    return np.concatenate(outputs)


def sweep_thresholds(probabilities, labels, thresholds=DEFAULT_THRESHOLDS):
    """Quét ngưỡng TRÊN DEV; trả bảng đầy đủ để báo cáo và 2 ngưỡng đề xuất."""
    rows = []
    for threshold in thresholds:
        score = metrics(labels, labels_from_probabilities(probabilities, threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": score["accuracy"],
                "macro_f1": score["macro"]["f1"],
                "toxic_precision": score["per_class"]["TOXIC"]["precision"],
                "toxic_recall": score["per_class"]["TOXIC"]["recall"],
                "toxic_f1": score["per_class"]["TOXIC"]["f1"],
            }
        )
    frame = pd.DataFrame(rows)
    return frame, {
        "best_macro_f1": float(frame.loc[frame.macro_f1.idxmax(), "threshold"]),
        "best_toxic_f1": float(frame.loc[frame.toxic_f1.idxmax(), "threshold"]),
        "default": 0.5,
    }


def evaluate_neural_test(state, *, confirm=False, tune_threshold=True):
    """Chấm BiLSTM/PhoBERT trên test. Ngưỡng chỉ được chọn bằng dev, không bằng test."""
    if not state.enabled:
        print("Bỏ qua đánh giá neural trên test vì nhánh neural không chạy.")
        return None
    run = state.run
    OUTPUT_DIR, RUN_MODE, SEED = run.output_dir, run.config.run_mode, run.config.seed
    assert confirm, "Review dev selection then explicitly confirm test"
    lock_path = OUTPUT_DIR / "selection.lock.json"
    assert lock_path.exists(), "Chưa khóa lựa chọn: chạy freeze_baseline trước"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["locked_before_test"], "Lock không hợp lệ"

    device = state.device
    test_df = load_vihsd_split(run.data_path, "test")
    if RUN_MODE == "SMOKE":
        test_df = stratified_sample(test_df, 400, SEED + 2)
    started = time.perf_counter()
    processed_test = SocialPreprocessor().transform(test_df.text)
    preprocessing_seconds = time.perf_counter() - started
    test_labels, dev_labels = test_df.label.tolist(), state.dev_df.label.tolist()

    sources = {
        "bilstm": OUTPUT_DIR / "bilstm_binary.best.pt",
        "phobert": OUTPUT_DIR / "phobert_binary.best",
    }
    results, threshold_reports = {}, {}
    for name, path in sources.items():
        if name not in state.payload or not path.exists():
            print(f"Bỏ qua {name} trên test: chưa có checkpoint tại {path.name}")
            continue
        started = time.perf_counter()
        if name == "bilstm":
            test_logits, _ = bilstm_logits(path, processed_test, device)
            dev_logits, _ = bilstm_logits(path, state.processed_dev, device)
        else:
            test_logits = phobert_logits(path, processed_test, device)
            dev_logits = phobert_logits(path, state.processed_dev, device)
        predict_seconds = time.perf_counter() - started
        test_probabilities = toxic_probabilities(test_logits)
        dev_probabilities = toxic_probabilities(dev_logits)

        entry = {
            "predict_seconds": predict_seconds,
            "best_epoch": state.payload[name]["best_epoch"],
            "default_threshold": 0.5,
            "test": metrics(test_labels, labels_from_probabilities(test_probabilities)),
        }
        if tune_threshold:
            sweep, suggested = sweep_thresholds(dev_probabilities, dev_labels)
            chosen = suggested["best_toxic_f1"]
            entry["tuned_threshold"] = {
                "value": chosen,
                "selected_on": "dev",
                "criterion": "toxic_f1",
                "dev": metrics(
                    dev_labels, labels_from_probabilities(dev_probabilities, chosen)
                ),
                "test": metrics(
                    test_labels, labels_from_probabilities(test_probabilities, chosen)
                ),
            }
            threshold_reports[name] = {"sweep": sweep, "suggested": suggested}
        results[name] = entry
        print(
            f"{name} • test Accuracy={entry['test']['accuracy']:.4f} | "
            f"Macro-F1={entry['test']['macro']['f1']:.4f} | "
            f"TOXIC Recall={entry['test']['per_class']['TOXIC']['recall']:.4f}",
            flush=True,
        )

    payload = {
        "run_id": run.run_id,
        "opened_after_lock": True,
        "selection_used_test": False,
        "threshold_selected_on": "dev" if tune_threshold else None,
        "refit_performed": False,
        "preprocessing_seconds": preprocessing_seconds,
        "test_rows": int(len(test_df)),
        "models": {
            name: {key: value for key, value in entry.items()} for name, entry in results.items()
        },
        "result_status": "reproduction_post_hoc",
    }
    write_json(OUTPUT_DIR / "neural_test_results.json", payload)
    for name, report in threshold_reports.items():
        report["sweep"].to_csv(
            OUTPUT_DIR / f"{name}_dev_threshold_sweep.csv",
            index=False,
            encoding="utf-8-sig",
        )
    state.payload["test_used_by_neural"] = True
    state.payload["neural_test_results"] = "neural_test_results.json"
    write_json(OUTPUT_DIR / "neural_results.json", state.payload)
    return NeuralTestResults(results, threshold_reports, test_df, payload)

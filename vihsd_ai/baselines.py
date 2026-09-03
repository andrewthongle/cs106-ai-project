"""Baseline training and frozen-test evaluation with explicit inputs/results."""

import json
import random
import time
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .artifacts import load_baseline, sha256_file, write_json
from .config import BINARY_LABELS, SOURCE_TO_BINARY, VIHSD_COMMIT, VIHSD_SHA256
from .data import load_vihsd_split, stratified_sample
from .metrics import metrics, rank_models
from .preprocessing import SocialPreprocessor


@dataclass
class BaselineResults:
    models: dict
    ranked: list
    selected_model: str


@dataclass
class Selection:
    manifest: dict
    dev_payload: dict
    lock: dict


@dataclass
class TestResults:
    model_name: str
    model: Any
    frame: pd.DataFrame
    prediction: Any
    metrics: dict
    predict_seconds: float
    lock: dict


def make_pipeline(classifier):
    return Pipeline(
        [
            ("preprocess", SocialPreprocessor()),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=60000,
                    sublinear_tf=True,
                    lowercase=False,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            ("classifier", classifier),
        ]
    )


def make_baselines(seed=42):
    """Return fresh estimator pipelines; callers may supply their own model dictionary."""
    return {
        "multinomial_nb": make_pipeline(MultinomialNB(alpha=0.5)),
        "logistic_regression": make_pipeline(
            LogisticRegression(
                C=2.0,
                max_iter=1500,
                class_weight="balanced",
                random_state=seed,
                solver="liblinear",
            )
        ),
        "linear_svc": make_pipeline(
            LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
        ),
    }


def train_baselines(train_df, dev_df, *, seed=42, models=None):
    random.seed(seed)
    np.random.seed(seed)
    models = make_baselines(seed) if models is None else models
    dev_rows = []
    for name, model in models.items():
        print(f"Đang fit {name}...", flush=True)
        started = time.perf_counter()
        model.fit(train_df.text.tolist(), train_df.label.tolist())
        fit_seconds = time.perf_counter() - started
        assert set(model.named_steps["classifier"].classes_) == {"SAFE", "TOXIC"}
        started = time.perf_counter()
        train_score = metrics(train_df.label, model.predict(train_df.text))
        dev_score = metrics(dev_df.label, model.predict(dev_df.text))
        eval_seconds = time.perf_counter() - started
        dev_rows.append(
            {
                "name": name,
                "fit_seconds": fit_seconds,
                "evaluation_seconds": eval_seconds,
                "train": train_score,
                "dev": dev_score,
            }
        )
        print(
            f"{name}: fit={fit_seconds:.1f}s | dev Macro-F1={dev_score['macro']['f1']:.4f} | "
            f"dev TOXIC Recall={dev_score['per_class']['TOXIC']['recall']:.4f}",
            flush=True,
        )
    ranked = rank_models(dev_rows)
    selected_model = ranked[0]["name"]
    print(f"Baseline được chọn để khóa/test/demo: {selected_model}")
    return BaselineResults(models, ranked, selected_model)


def freeze_baseline(run, baselines):
    selected_model, ranked = baselines.selected_model, baselines.ranked
    OUTPUT_DIR = run.output_dir
    RUN_ID, RUN_MODE, SEED, NEURAL_EPOCHS = (
        run.run_id,
        run.config.run_mode,
        run.config.seed,
        run.config.neural_epochs,
    )
    artifact_path = OUTPUT_DIR / f"{selected_model}.joblib"
    joblib.dump(baselines.models[selected_model], artifact_path, compress=3)
    dev_payload = {
        "test_consulted": False,
        "training_targets": BINARY_LABELS,
        "selection_rule": ["dev_macro_f1", "dev_toxic_recall"],
        "selected_model": selected_model,
        "models": ranked,
    }
    write_json(OUTPUT_DIR / "dev_results.json", dev_payload)
    manifest = {
        "run_id": RUN_ID,
        "seed": SEED,
        "neural_epochs": NEURAL_EPOCHS,
        "run_mode": RUN_MODE,
        "dataset_commit": VIHSD_COMMIT,
        "dataset_sha256": VIHSD_SHA256,
        "label_mapping_before_fit": SOURCE_TO_BINARY,
        "test_loaded_during_training": False,
        "result_status": "reproduction_post_hoc",
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    lock = {
        "locked": True,
        "locked_before_test": True,
        "test_opened": False,
        "selected_model": selected_model,
        "artifact": {"path": artifact_path.name, "sha256": sha256_file(artifact_path)},
        "dev_results": {
            "path": "dev_results.json",
            "sha256": sha256_file(OUTPUT_DIR / "dev_results.json"),
        },
    }
    write_json(OUTPUT_DIR / "selection.lock.json", lock)

    return Selection(manifest, dev_payload, lock)


def evaluate_test(run, *, confirm=False):
    """Verify the frozen artifacts before the first read of test."""
    OUTPUT_DIR, RUN_ID = run.output_dir, run.run_id
    RUN_MODE, SEED = run.config.run_mode, run.config.seed
    CONFIRM_TEST_AFTER_LOCK = confirm
    assert CONFIRM_TEST_AFTER_LOCK, "Review dev selection then explicitly confirm test"
    lock = json.loads((OUTPUT_DIR / "selection.lock.json").read_text(encoding="utf-8"))
    assert lock["locked_before_test"] and not lock["test_opened"]
    assert (
        sha256_file(OUTPUT_DIR / lock["artifact"]["path"]) == lock["artifact"]["sha256"]
    )
    assert (
        sha256_file(OUTPUT_DIR / lock["dev_results"]["path"])
        == lock["dev_results"]["sha256"]
    )
    test_df = load_vihsd_split(
        run.data_path, "test"
    )  # first test read occurs after lock
    if RUN_MODE == "SMOKE":
        test_df = stratified_sample(test_df, 400, SEED + 2)
    frozen_model = load_baseline(OUTPUT_DIR / lock["artifact"]["path"])
    test_started = time.perf_counter()
    test_prediction = frozen_model.predict(test_df.text.tolist())
    test_predict_seconds = time.perf_counter() - test_started
    test_metrics = metrics(test_df.label.tolist(), test_prediction)
    test_payload = {
        "selected_model": lock["selected_model"],
        "predict_seconds": test_predict_seconds,
        "run_id": RUN_ID,
        "refit_performed": False,
        "selection_performed_during_test": False,
        "test_used_for_fit_or_selection": False,
        "metrics": test_metrics,
        "result_status": "reproduction_post_hoc",
    }
    write_json(OUTPUT_DIR / "test_results.json", test_payload)
    lock["test_opened"] = True
    lock["test_result_sha256"] = sha256_file(OUTPUT_DIR / "test_results.json")
    write_json(OUTPUT_DIR / "selection.lock.json", lock)

    return TestResults(
        lock["selected_model"],
        frozen_model,
        test_df,
        test_prediction,
        test_metrics,
        test_predict_seconds,
        lock,
    )

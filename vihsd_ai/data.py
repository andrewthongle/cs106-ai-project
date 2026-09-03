"""Verified dataset loading, binary labels, sampling and aggregate-only audit."""

import hashlib
import urllib.request
import zipfile

import pandas as pd

from .artifacts import sha256_file
from .config import (
    BINARY_LABELS,
    ID_TO_SOURCE,
    SOURCE_TO_BINARY,
    VIHSD_SHA256,
    VIHSD_URL,
)


def prepare_dataset(run):
    """Download/check the archive without opening any split."""
    if not run.data_path.exists() or sha256_file(run.data_path) != VIHSD_SHA256:
        run.data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VIHSD_URL, run.data_path)
    assert sha256_file(run.data_path) == VIHSD_SHA256, (
        "Checksum mismatch: stop before reading data"
    )
    print("ViHSD checksum verified")
    print(f"Thư mục kết quả lần chạy này: {run.output_dir.resolve()}")


def load_vihsd_split(data_path, split):
    with (
        zipfile.ZipFile(data_path) as archive,
        archive.open(f"vihsd/{split}.csv") as stream,
    ):
        raw = pd.read_csv(stream)
    source = raw["label_id"].astype(int).map(ID_TO_SOURCE)
    if source.isna().any():
        raise ValueError("Unknown ViHSD ground-truth label")
    frame = pd.DataFrame(
        {"text": raw["free_text"].fillna("").astype(str), "source_label": source}
    )
    frame["label"] = frame["source_label"].map(SOURCE_TO_BINARY)  # remap BEFORE fit
    assert set(frame["label"]) <= set(BINARY_LABELS)
    return frame


def stratified_sample(frame, size, seed):
    if len(frame) <= size:
        return frame.reset_index(drop=True)
    pieces = []
    for offset, (_, group) in enumerate(frame.groupby("label")):
        count = max(2, round(size * len(group) / len(frame)))
        pieces.append(
            group.sample(n=min(count, len(group)), random_state=seed + offset)
        )
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sample(n=size, random_state=seed).reset_index(drop=True)


def load_train_dev(run):
    train = load_vihsd_split(run.data_path, "train")
    dev = load_vihsd_split(run.data_path, "dev")
    if run.config.run_mode == "SMOKE":
        train = stratified_sample(train, 600, run.config.seed)
        dev = stratified_sample(dev, 240, run.config.seed + 1)
    return train, dev


def fingerprint(text):
    return hashlib.sha256(
        b"vihsd-ai-notebook-v1\0" + text.encode("utf-8", errors="replace")
    ).hexdigest()


def private_audit(frame):
    lengths = frame.text.str.len()
    unique = len({fingerprint(text) for text in frame.text})
    return {
        "rows": len(frame),
        "binary_distribution": frame.label.value_counts().to_dict(),
        "source_distribution": frame.source_label.value_counts().to_dict(),
        "length": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": float(lengths.mean()),
            "p95": float(lengths.quantile(0.95)),
            "max": int(lengths.max()),
        },
        "contains_url": int(
            frame.text.str.contains(r"https?://|www\.", case=False, regex=True).sum()
        ),
        "empty_texts": int(frame.text.str.strip().eq("").sum()),
        "unique_fingerprints": unique,
        "duplicate_rows": len(frame) - unique,
    }

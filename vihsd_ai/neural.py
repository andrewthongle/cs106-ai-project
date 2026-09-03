"""BiLSTM/PhoBERT training on train/dev only. Heavy imports are lazy."""

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .artifacts import write_json
from .config import BINARY_LABELS, RunContext
from .metrics import metrics, selection_key
from .preprocessing import SocialPreprocessor


@dataclass
class NeuralRun:
    run: RunContext
    train_df: pd.DataFrame
    dev_df: pd.DataFrame
    payload: dict
    history: list = field(default_factory=list)
    device: Any = None
    processed_train: list = field(default_factory=list)
    processed_dev: list = field(default_factory=list)

    @property
    def enabled(self):
        return self.run.config.run_mode == "FULL_WITH_NEURAL"


def history_frame(rows):
    return pd.DataFrame(
        [
            {
                "model": row["model"],
                "epoch": row["epoch"],
                "optimization_loss": row["optimization_loss"],
                "train_loss": row["train_loss"],
                "dev_loss": row["dev_loss"],
                "train_macro_f1": row["train"]["macro"]["f1"],
                "dev_macro_f1": row["dev"]["macro"]["f1"],
                "dev_accuracy": row["dev"]["accuracy"],
                "dev_toxic_recall": row["dev"]["per_class"]["TOXIC"]["recall"],
                "fit_seconds": row["fit_seconds"],
                "evaluation_seconds": row["evaluation_seconds"],
                "epoch_seconds": row["epoch_seconds"],
            }
            for row in rows
        ]
    )


def persist_history(state, row):
    state.history.append(row)
    write_json(
        state.run.output_dir / "neural_history.json",
        {"run_id": state.run.run_id, "epochs": state.history},
    )
    history_frame(state.history).to_csv(
        state.run.output_dir / "neural_history.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"{row['model']} epoch {row['epoch']}/{state.run.config.neural_epochs} | "
        f"train loss={row['train_loss']:.4f}, dev loss={row['dev_loss']:.4f} | "
        f"train F1={row['train']['macro']['f1']:.4f}, dev F1={row['dev']['macro']['f1']:.4f} | "
        f"dev TOXIC Recall={row['dev']['per_class']['TOXIC']['recall']:.4f} | "
        f"{row['epoch_seconds']:.1f}s",
        flush=True,
    )


def prepare_neural(run, train_df, dev_df):
    state = NeuralRun(
        run,
        train_df,
        dev_df,
        {
            "run_id": run.run_id,
            "selection_split": "dev",
            "test_used_by_neural": False,
            "status": "running"
            if run.config.run_mode == "FULL_WITH_NEURAL"
            else "not_requested",
        },
    )
    if not state.enabled:
        print(
            f"Bỏ qua BiLSTM/PhoBERT vì RUN_MODE={run.config.run_mode}. Bảng tổng hợp sẽ chỉ có 3 baseline."
        )
        return state
    import torch

    torch.manual_seed(run.config.seed)
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if state.device.type == "cuda":
        torch.cuda.manual_seed_all(run.config.seed)
    print(
        f"Neural device: {state.device}"
        + (
            f" • {torch.cuda.get_device_name(0)}"
            if state.device.type == "cuda"
            else " • CPU sẽ chạy chậm"
        ),
        flush=True,
    )
    started = time.perf_counter()
    state.processed_train = SocialPreprocessor().transform(train_df.text)
    state.processed_dev = SocialPreprocessor().transform(dev_df.text)
    state.payload.update(
        device=str(state.device), preprocessing_seconds=time.perf_counter() - started
    )
    write_json(run.output_dir / "neural_results.json", state.payload)
    return state


def train_bilstm(state, *, learning_rate=2e-4):
    if not state.enabled:
        return None
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    train_df, dev_df = state.train_df, state.dev_df
    processed_train, processed_dev = state.processed_train, state.processed_dev
    device, OUTPUT_DIR = state.device, state.run.output_dir
    NEURAL_EPOCHS = state.run.config.neural_epochs
    neural_payload, neural_history = state.payload, state.history
    model_started = time.perf_counter()
    counts = Counter(token for text in processed_train for token in text.split())
    vocab_tokens = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ][:39998]
    vocabulary = {
        "<pad>": 0,
        "<unk>": 1,
        **{token: i + 2 for i, token in enumerate(vocab_tokens)},
    }

    def encode(texts, max_length=128):
        rows = []
        for text in texts:
            row = [vocabulary.get(token, 1) for token in text.split()[:max_length]]
            rows.append(row + [0] * (max_length - len(row)))
        return torch.tensor(rows, dtype=torch.long)

    class BinaryBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(vocabulary), 128, padding_idx=0)
            self.lstm = nn.LSTM(128, 96, batch_first=True, bidirectional=True)
            self.head = nn.Linear(192, 2)

        def forward(self, x):
            _, (h, _) = self.lstm(self.emb(x))
            return self.head(torch.cat((h[-2], h[-1]), dim=1))

    bilstm = BinaryBiLSTM().to(device)
    optimizer = torch.optim.AdamW(bilstm.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_x, dev_x = encode(processed_train), encode(processed_dev)
    train_y = torch.tensor(
        (train_df.label == "TOXIC").astype(int).to_numpy(), dtype=torch.long
    )
    dev_y = torch.tensor(
        (dev_df.label == "TOXIC").astype(int).to_numpy(), dtype=torch.long
    )
    train_set, dev_set = TensorDataset(train_x, train_y), TensorDataset(dev_x, dev_y)
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    train_eval_loader = DataLoader(train_set, batch_size=64)
    dev_eval_loader = DataLoader(dev_set, batch_size=64)

    def evaluate_bilstm(model, eval_loader, labels):
        model.eval()
        total_loss, total_rows, predictions = 0.0, 0, []
        with torch.inference_mode():
            for x, y in eval_loader:
                logits = model(x.to(device))
                total_loss += float(loss_fn(logits, y.to(device)).item()) * len(y)
                total_rows += len(y)
                predictions.extend(logits.argmax(1).cpu().tolist())
        return total_loss / total_rows, metrics(
            labels, [BINARY_LABELS[i] for i in predictions]
        )

    best_bilstm = None
    for epoch in range(1, NEURAL_EPOCHS + 1):
        epoch_started = time.perf_counter()
        bilstm.train()
        running_loss = 0.0
        for batch_index, (x, y) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = loss_fn(bilstm(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(y)
            if batch_index % 250 == 0:
                print(
                    f"BiLSTM epoch {epoch}: batch {batch_index}/{len(loader)}",
                    flush=True,
                )
        fit_seconds = time.perf_counter() - epoch_started
        eval_started = time.perf_counter()
        train_loss, train_score = evaluate_bilstm(
            bilstm, train_eval_loader, train_df.label.tolist()
        )
        dev_loss, dev_score = evaluate_bilstm(
            bilstm, dev_eval_loader, dev_df.label.tolist()
        )
        row = {
            "model": "bilstm",
            "epoch": epoch,
            "optimization_loss": running_loss / len(train_df),
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "train": train_score,
            "dev": dev_score,
            "fit_seconds": fit_seconds,
            "evaluation_seconds": time.perf_counter() - eval_started,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        persist_history(state, row)
        if best_bilstm is None or selection_key(dev_score) > selection_key(
            best_bilstm["dev"]
        ):
            best_bilstm = row
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in bilstm.state_dict().items()
                    },
                    "vocabulary": vocabulary,
                    "labels": BINARY_LABELS,
                    "best_epoch": epoch,
                },
                OUTPUT_DIR / "bilstm_binary.best.pt",
            )
    neural_payload["bilstm"] = {
        "best_epoch": best_bilstm["epoch"],
        "train": best_bilstm["train"],
        "dev": best_bilstm["dev"],
        "fit_seconds": sum(
            row["fit_seconds"] for row in neural_history if row["model"] == "bilstm"
        ),
        "evaluation_seconds": sum(
            row["evaluation_seconds"]
            for row in neural_history
            if row["model"] == "bilstm"
        ),
        "total_seconds": time.perf_counter() - model_started,
        "vocabulary_size": len(vocabulary),
        "max_length": 128,
    }
    write_json(OUTPUT_DIR / "neural_results.json", neural_payload)
    print(f"BiLSTM: đã lưu checkpoint tốt nhất ở epoch {best_bilstm['epoch']}")
    # Giải phóng GPU trước PhoBERT; checkpoint đã lưu trên đĩa.
    del optimizer, bilstm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state.payload["bilstm"]


def train_phobert(state, *, learning_rate=2e-5):
    if not state.enabled:
        return None
    import torch

    train_df, dev_df = state.train_df, state.dev_df
    processed_train, processed_dev = state.processed_train, state.processed_dev
    device, OUTPUT_DIR = state.device, state.run.output_dir
    NEURAL_EPOCHS = state.run.config.neural_epochs
    neural_payload, neural_history = state.payload, state.history
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_started = time.perf_counter()
    checkpoint = "vinai/phobert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision="main")
    phobert = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        revision="main",
        num_labels=2,
        id2label={0: "SAFE", 1: "TOXIC"},
        label2id={"SAFE": 0, "TOXIC": 1},
    ).to(device)
    optimizer = torch.optim.AdamW(phobert.parameters(), lr=learning_rate)

    def batches(texts, labels, batch_size=16):
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            encoded["labels"] = torch.tensor(
                [
                    1 if label == "TOXIC" else 0
                    for label in labels[start : start + batch_size]
                ],
                device=device,
            )
            yield encoded

    def evaluate_phobert(model, texts, labels):
        model.eval()
        total_loss, predictions = 0.0, []
        with torch.inference_mode():
            for batch in batches(texts, labels):
                output = model(**batch)
                total_loss += float(output.loss.item()) * len(batch["labels"])
                predictions.extend(output.logits.argmax(1).cpu().tolist())
        return total_loss / len(labels), metrics(
            labels, [BINARY_LABELS[i] for i in predictions]
        )

    best_phobert = None
    train_labels, dev_labels = train_df.label.tolist(), dev_df.label.tolist()
    for epoch in range(1, NEURAL_EPOCHS + 1):
        epoch_started = time.perf_counter()
        phobert.train()
        running_loss = 0.0
        for batch_index, batch in enumerate(batches(processed_train, train_labels), 1):
            optimizer.zero_grad()
            output = phobert(**batch)
            output.loss.backward()
            optimizer.step()
            running_loss += float(output.loss.item()) * len(batch["labels"])
            if batch_index % 250 == 0:
                print(
                    f"PhoBERT epoch {epoch}: batch {batch_index}/{math.ceil(len(train_labels) / 16)}",
                    flush=True,
                )
        fit_seconds = time.perf_counter() - epoch_started
        eval_started = time.perf_counter()
        train_loss, train_score = evaluate_phobert(
            phobert, processed_train, train_labels
        )
        dev_loss, dev_score = evaluate_phobert(phobert, processed_dev, dev_labels)
        row = {
            "model": "phobert",
            "epoch": epoch,
            "optimization_loss": running_loss / len(train_df),
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "train": train_score,
            "dev": dev_score,
            "fit_seconds": fit_seconds,
            "evaluation_seconds": time.perf_counter() - eval_started,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        persist_history(state, row)
        if best_phobert is None or selection_key(dev_score) > selection_key(
            best_phobert["dev"]
        ):
            best_phobert = row
            phobert.save_pretrained(OUTPUT_DIR / "phobert_binary.best")
            tokenizer.save_pretrained(OUTPUT_DIR / "phobert_binary.best")
    neural_payload["phobert"] = {
        "best_epoch": best_phobert["epoch"],
        "train": best_phobert["train"],
        "dev": best_phobert["dev"],
        "fit_seconds": sum(
            row["fit_seconds"] for row in neural_history if row["model"] == "phobert"
        ),
        "evaluation_seconds": sum(
            row["evaluation_seconds"]
            for row in neural_history
            if row["model"] == "phobert"
        ),
        "total_seconds": time.perf_counter() - model_started,
        "checkpoint": checkpoint,
        "resolved_revision": getattr(phobert.config, "_commit_hash", None),
        "max_length": 128,
    }
    neural_payload["status"] = "completed"
    write_json(OUTPUT_DIR / "neural_results.json", neural_payload)
    print(f"PhoBERT: đã lưu checkpoint tốt nhất ở epoch {best_phobert['epoch']}")
    del optimizer, phobert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state.payload["phobert"]

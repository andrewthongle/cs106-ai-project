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


def class_weights(labels, device=None):
    """Trọng số cân bằng lớp theo công thức của sklearn: n / (số lớp * số mẫu lớp đó)."""
    import torch

    counts = Counter(labels)
    total, classes = len(labels), len(BINARY_LABELS)
    weights = [total / (classes * counts[label]) for label in BINARY_LABELS]
    return torch.tensor(weights, dtype=torch.float, device=device), dict(
        zip(BINARY_LABELS, weights)
    )


def batch_weight(targets, weight_tensor):
    """Mẫu số đúng để khử trung bình khi cộng dồn loss có trọng số lớp.

    CrossEntropyLoss(weight=..., reduction="mean") chia tổng loss cho TỔNG TRỌNG SỐ
    của batch, không phải cho số mẫu. Nhân lại với len(batch) sẽ khử sai mẫu số và
    làm lệch con số loss báo cáo.
    """
    if weight_tensor is None:
        return float(len(targets))
    return float(weight_tensor[targets].sum().item())


def build_vocabulary(processed_texts, *, min_count=2, max_size=39998):
    counts = Counter(token for text in processed_texts for token in text.split())
    vocab_tokens = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_count
    ][:max_size]
    return {
        "<pad>": 0,
        "<unk>": 1,
        **{token: i + 2 for i, token in enumerate(vocab_tokens)},
    }


def pretrained_embedding_matrix(
    processed_train, vocabulary, *, embedding_dim, seed, min_count=2, epochs=5
):
    """Word2Vec skip-gram huấn luyện trên chính tập train; trả về (ma trận, thống kê).

    Trả (None, thông tin lỗi) khi thiếu gensim để nhánh gọi tự lùi về khởi tạo ngẫu nhiên.
    """
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        return None, {"used": False, "reason": f"gensim không khả dụng ({exc})"}
    import numpy as np

    started = time.perf_counter()
    sentences = [text.split() for text in processed_train]
    word2vec = Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=5,
        min_count=min_count,
        sg=1,
        seed=seed,
        workers=1,  # bắt buộc để tái lập được kết quả
        epochs=epochs,
    )
    generator = np.random.default_rng(seed)
    matrix = generator.normal(0.0, 0.1, size=(len(vocabulary), embedding_dim)).astype(
        "float32"
    )
    matrix[0] = 0.0  # <pad>
    covered = 0
    for token, index in vocabulary.items():
        if token in word2vec.wv:
            matrix[index] = word2vec.wv[token]
            covered += 1
    return matrix, {
        "used": True,
        "algorithm": "word2vec_skipgram",
        "corpus": "train split đã tiền xử lý",
        "vector_size": embedding_dim,
        "window": 5,
        "min_count": min_count,
        "epochs": epochs,
        "seed": seed,
        "vocabulary_covered": covered,
        "vocabulary_size": len(vocabulary),
        "coverage_ratio": covered / len(vocabulary),
        "train_seconds": time.perf_counter() - started,
    }


def encode_texts(texts, vocabulary, max_length):
    """Trả (ids đã pad, độ dài thật). Độ dài tối thiểu 1 để pack_padded_sequence không lỗi."""
    import torch

    rows, lengths = [], []
    for text in texts:
        row = [vocabulary.get(token, 1) for token in text.split()[:max_length]]
        lengths.append(max(len(row), 1))
        rows.append(row + [0] * (max_length - len(row)))
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def make_bilstm(vocabulary_size, embedding_dim, hidden_size, dropout):
    """Nhà máy tạo model; torch chỉ được nạp khi thật sự chạy nhánh neural.

    Dùng chung cho lúc train và lúc nạp lại checkpoint để đánh giá, tránh hai
    định nghĩa kiến trúc bị lệch nhau.
    """
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    class BinaryBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embedding_dim, hidden_size, batch_first=True, bidirectional=True
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size * 2, 2)

        def forward(self, x, lengths):
            # Pack để LSTM dừng đúng token cuối; nếu không, hidden state chiều xuôi
            # sẽ được lấy tại vị trí padding và biểu diễn câu bị hỏng.
            packed = pack_padded_sequence(
                self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h, _) = self.lstm(packed)
            return self.head(self.dropout(torch.cat((h[-2], h[-1]), dim=1)))

    return BinaryBiLSTM()


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


def train_bilstm(
    state,
    *,
    learning_rate=2e-4,
    batch_size=32,
    max_length=128,
    embedding_dim=128,
    hidden_size=96,
    dropout=0.3,
    min_count=2,
    weight_decay=0.01,
    patience=2,
    grad_clip=1.0,
    use_class_weights=True,
    use_pretrained_embeddings=True,
):
    if not state.enabled:
        return None
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    train_df, dev_df = state.train_df, state.dev_df
    processed_train, processed_dev = state.processed_train, state.processed_dev
    device, OUTPUT_DIR = state.device, state.run.output_dir
    NEURAL_EPOCHS, SEED = state.run.config.neural_epochs, state.run.config.seed
    neural_payload, neural_history = state.payload, state.history
    model_started = time.perf_counter()
    vocabulary = build_vocabulary(processed_train, min_count=min_count)

    embedding_matrix, embedding_info = (None, {"used": False, "reason": "đã tắt"})
    if use_pretrained_embeddings:
        embedding_matrix, embedding_info = pretrained_embedding_matrix(
            processed_train,
            vocabulary,
            embedding_dim=embedding_dim,
            seed=SEED,
            min_count=min_count,
        )
    if embedding_info["used"]:
        print(
            f"BiLSTM: nhúng từ Word2Vec phủ {embedding_info['vocabulary_covered']}/"
            f"{embedding_info['vocabulary_size']} từ vựng "
            f"({embedding_info['coverage_ratio']:.1%}), {embedding_info['train_seconds']:.1f}s",
            flush=True,
        )
    else:
        print(
            f"BiLSTM: dùng nhúng ngẫu nhiên — {embedding_info['reason']}",
            flush=True,
        )

    def encode(texts):
        return encode_texts(texts, vocabulary, max_length)

    bilstm = make_bilstm(len(vocabulary), embedding_dim, hidden_size, dropout).to(device)
    if embedding_matrix is not None:
        with torch.no_grad():
            bilstm.emb.weight.copy_(torch.from_numpy(embedding_matrix))
            bilstm.emb.weight[0].zero_()
    optimizer = torch.optim.AdamW(
        bilstm.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    weight_tensor, weight_map = (None, None)
    if use_class_weights:
        weight_tensor, weight_map = class_weights(train_df.label.tolist(), device)
        print(
            "BiLSTM: trọng số lớp "
            + ", ".join(f"{k}={v:.3f}" for k, v in weight_map.items()),
            flush=True,
        )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    train_x, train_lengths = encode(processed_train)
    dev_x, dev_lengths = encode(processed_dev)
    train_y = torch.tensor(
        (train_df.label == "TOXIC").astype(int).to_numpy(), dtype=torch.long
    )
    dev_y = torch.tensor(
        (dev_df.label == "TOXIC").astype(int).to_numpy(), dtype=torch.long
    )
    train_set = TensorDataset(train_x, train_lengths, train_y)
    dev_set = TensorDataset(dev_x, dev_lengths, dev_y)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_set, batch_size=batch_size * 2)
    dev_eval_loader = DataLoader(dev_set, batch_size=batch_size * 2)

    def evaluate_bilstm(model, eval_loader, labels):
        model.eval()
        total_loss, total_weight, predictions = 0.0, 0.0, []
        with torch.inference_mode():
            for x, lengths, y in eval_loader:
                y = y.to(device)
                logits = model(x.to(device), lengths)
                weight = batch_weight(y, weight_tensor)
                total_loss += float(loss_fn(logits, y).item()) * weight
                total_weight += weight
                predictions.extend(logits.argmax(1).cpu().tolist())
        return total_loss / total_weight, metrics(
            labels, [BINARY_LABELS[i] for i in predictions]
        )

    best_bilstm, stale_epochs, stopped_early = None, 0, False
    for epoch in range(1, NEURAL_EPOCHS + 1):
        epoch_started = time.perf_counter()
        bilstm.train()
        running_loss, running_weight = 0.0, 0.0
        for batch_index, (x, lengths, y) in enumerate(loader, 1):
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(bilstm(x.to(device), lengths), y)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(bilstm.parameters(), grad_clip)
            optimizer.step()
            weight = batch_weight(y, weight_tensor)
            running_loss += float(loss.item()) * weight
            running_weight += weight
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
            "optimization_loss": running_loss / running_weight,
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
            best_bilstm, stale_epochs = row, 0
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in bilstm.state_dict().items()
                    },
                    "vocabulary": vocabulary,
                    "labels": BINARY_LABELS,
                    "best_epoch": epoch,
                    "architecture": {
                        "embedding_dim": embedding_dim,
                        "hidden_size": hidden_size,
                        "dropout": dropout,
                        "max_length": max_length,
                        "vocabulary_size": len(vocabulary),
                    },
                },
                OUTPUT_DIR / "bilstm_binary.best.pt",
            )
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                stopped_early = True
                print(
                    f"BiLSTM: dừng sớm ở epoch {epoch} — dev không cải thiện {stale_epochs} epoch liên tiếp",
                    flush=True,
                )
                break
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
        "max_length": max_length,
        "epochs_ran": sum(1 for row in neural_history if row["model"] == "bilstm"),
        "stopped_early": stopped_early,
        "embedding": embedding_info,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_length": max_length,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "min_count": min_count,
            "weight_decay": weight_decay,
            "patience": patience,
            "grad_clip": grad_clip,
            "packed_sequence": True,
            "class_weights": weight_map,
        },
    }
    write_json(OUTPUT_DIR / "neural_results.json", neural_payload)
    print(f"BiLSTM: đã lưu checkpoint tốt nhất ở epoch {best_bilstm['epoch']}")
    # Giải phóng GPU trước PhoBERT; checkpoint đã lưu trên đĩa.
    del optimizer, bilstm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state.payload["bilstm"]


def train_phobert(
    state,
    *,
    learning_rate=2e-5,
    batch_size=16,
    max_length=128,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=2,
    grad_clip=1.0,
    use_class_weights=True,
):
    if not state.enabled:
        return None
    import torch
    from torch import nn

    train_df, dev_df = state.train_df, state.dev_df
    processed_train, processed_dev = state.processed_train, state.processed_dev
    device, OUTPUT_DIR = state.device, state.run.output_dir
    NEURAL_EPOCHS = state.run.config.neural_epochs
    neural_payload, neural_history = state.payload, state.history
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

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
    optimizer = torch.optim.AdamW(
        phobert.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    steps_per_epoch = math.ceil(len(processed_train) / batch_size)
    total_steps = steps_per_epoch * NEURAL_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    weight_tensor, weight_map = (None, None)
    if use_class_weights:
        weight_tensor, weight_map = class_weights(train_df.label.tolist(), device)
        print(
            "PhoBERT: trọng số lớp "
            + ", ".join(f"{k}={v:.3f}" for k, v in weight_map.items()),
            flush=True,
        )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    def batches(texts, labels):
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            targets = torch.tensor(
                [
                    1 if label == "TOXIC" else 0
                    for label in labels[start : start + batch_size]
                ],
                device=device,
            )
            yield encoded, targets

    def evaluate_phobert(model, texts, labels):
        model.eval()
        total_loss, total_weight, predictions = 0.0, 0.0, []
        with torch.inference_mode():
            for encoded, targets in batches(texts, labels):
                logits = model(**encoded).logits
                weight = batch_weight(targets, weight_tensor)
                total_loss += float(loss_fn(logits, targets).item()) * weight
                total_weight += weight
                predictions.extend(logits.argmax(1).cpu().tolist())
        return total_loss / total_weight, metrics(
            labels, [BINARY_LABELS[i] for i in predictions]
        )

    best_phobert, stale_epochs, stopped_early = None, 0, False
    train_labels, dev_labels = train_df.label.tolist(), dev_df.label.tolist()
    for epoch in range(1, NEURAL_EPOCHS + 1):
        epoch_started = time.perf_counter()
        phobert.train()
        running_loss, running_weight = 0.0, 0.0
        for batch_index, (encoded, targets) in enumerate(
            batches(processed_train, train_labels), 1
        ):
            optimizer.zero_grad()
            # Tự tính loss thay vì để model tính, để áp được trọng số lớp.
            loss = loss_fn(phobert(**encoded).logits, targets)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(phobert.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            weight = batch_weight(targets, weight_tensor)
            running_loss += float(loss.item()) * weight
            running_weight += weight
            if batch_index % 250 == 0:
                print(
                    f"PhoBERT epoch {epoch}: batch {batch_index}/{steps_per_epoch}",
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
            "optimization_loss": running_loss / running_weight,
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
            best_phobert, stale_epochs = row, 0
            phobert.save_pretrained(OUTPUT_DIR / "phobert_binary.best")
            tokenizer.save_pretrained(OUTPUT_DIR / "phobert_binary.best")
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                stopped_early = True
                print(
                    f"PhoBERT: dừng sớm ở epoch {epoch} — dev không cải thiện {stale_epochs} epoch liên tiếp",
                    flush=True,
                )
                break
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
        "max_length": max_length,
        "epochs_ran": sum(1 for row in neural_history if row["model"] == "phobert"),
        "stopped_early": stopped_early,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_length": max_length,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "patience": patience,
            "grad_clip": grad_clip,
            "scheduler": "linear_warmup_decay",
            "class_weights": weight_map,
        },
    }
    neural_payload["status"] = "completed"
    write_json(OUTPUT_DIR / "neural_results.json", neural_payload)
    print(f"PhoBERT: đã lưu checkpoint tốt nhất ở epoch {best_phobert['epoch']}")
    del optimizer, phobert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state.payload["phobert"]

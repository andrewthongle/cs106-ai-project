"""Notebook-friendly tables/figures and CSV/PNG exports, separate from training."""

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

from .artifacts import write_json
from .config import BINARY_LABELS
from .data import private_audit
from .neural import history_frame

COLORS = {"SAFE": "#2878B5", "TOXIC": "#D9534F", "train": "#2878B5", "dev": "#E69F00"}


def annotate_bars(ax, bars, percentage=False, decimals=0):
    for bar in bars:
        value = bar.get_height()
        if np.isfinite(value):
            label = (
                (f"{100 * value:.1f}".rstrip("0").rstrip(".") + "%")
                if percentage
                else f"{value:,.{decimals}f}"
            )
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


class Reports:
    """One report writer per experiment; independent runs never share output paths."""

    def __init__(self, run):
        self.run = run
        self.output_dir = Path(run.output_dir)
        plt.rcParams.update(
            {
                "figure.dpi": 110,
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    def show_table(self, frame, title, filename=None):
        """Hiện đủ hàng/cột và toàn bộ nội dung; cuộn khi bảng rộng/dài."""
        print(f"\n{title}")
        with pd.option_context("display.max_colwidth", None):
            table_html = frame.to_html(
                index=False,
                max_rows=None,
                max_cols=None,
                escape=True,
                float_format=lambda value: f"{value:.4f}",
                na_rep="—",
            )
        display(
            HTML('<div style="overflow:auto;max-height:650px">' + table_html + "</div>")
        )
        if filename:
            frame.to_csv(self.output_dir / filename, index=False, encoding="utf-8-sig")

    def show_figure(self, fig, filename):
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_confusions(self, entries, filename, title):
        """Mỗi model/split: số lượng và phần trăm chuẩn hóa theo nhãn thật."""
        fig, axes = plt.subplots(
            len(entries), 2, figsize=(10, 3.5 * len(entries)), squeeze=False
        )
        for row, (label, score) in enumerate(entries):
            cm = np.asarray(score["confusion_matrix"]["rows_true_columns_predicted"])
            totals = cm.sum(axis=1, keepdims=True)
            normalized = np.divide(
                cm, totals, out=np.zeros_like(cm, dtype=float), where=totals != 0
            )
            for col, values in enumerate((cm, normalized)):
                ax = axes[row, col]
                vmax = max(float(values.max()), 1) if col == 0 else 1
                ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
                ax.set(
                    xticks=range(2),
                    yticks=range(2),
                    xticklabels=BINARY_LABELS,
                    yticklabels=BINARY_LABELS,
                    xlabel="Dự đoán",
                    ylabel="Nhãn thật",
                    title=f"{label} • {'Số lượng' if col == 0 else '% theo nhãn thật'}",
                )
                for i in range(2):
                    for j in range(2):
                        text = (
                            f"{cm[i, j]:,}"
                            if col == 0
                            else (f"{values[i, j]:.1%}" if totals[i, 0] else "N/A")
                        )
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color="white" if values[i, j] > vmax / 2 else "black",
                        )
        fig.suptitle(title, fontsize=13)
        self.show_figure(fig, filename)

    def metric_tables(self, entries, prefix):
        summary, per_class = [], []
        for name, split, score in entries:
            summary.append(
                {
                    "model": name,
                    "split": split,
                    "samples": sum(v["support"] for v in score["per_class"].values()),
                    "accuracy": score["accuracy"],
                    **{
                        f"macro_{k}": score["macro"][k]
                        for k in ("precision", "recall", "f1")
                    },
                    **{
                        f"weighted_{k}": score["weighted"][k]
                        for k in ("precision", "recall", "f1")
                    },
                }
            )
            for label in BINARY_LABELS:
                per_class.append(
                    {
                        "model": name,
                        "split": split,
                        "label": label,
                        **score["per_class"][label],
                    }
                )
        self.show_table(
            pd.DataFrame(summary),
            "Metrics tổng hợp (thang 0–1)",
            f"{prefix}_summary.csv",
        )
        self.show_table(
            pd.DataFrame(per_class),
            "Precision / Recall / F1 / Support từng lớp",
            f"{prefix}_per_class.csv",
        )

    def display_audit(self, frames, filename_prefix):
        stats = {name: private_audit(frame) for name, frame in frames.items()}
        rows = []
        for split, values in stats.items():
            rows.append(
                {
                    "split": split,
                    **{k: v for k, v in values.items() if not isinstance(v, dict)},
                    **{f"length_{k}": v for k, v in values["length"].items()},
                }
            )
        self.show_table(
            pd.DataFrame(rows),
            "Thống kê tổng hợp; độ dài tính bằng ký tự",
            f"{filename_prefix}_stats.csv",
        )
        distribution = [
            {
                "split": split,
                "label_type": kind,
                "label": label,
                "count": int(count),
                "ratio": count / values["rows"],
            }
            for split, values in stats.items()
            for kind in ("source_distribution", "binary_distribution")
            for label, count in values[kind].items()
        ]
        self.show_table(
            pd.DataFrame(distribution),
            "Phân bố nhãn gốc và nhãn nhị phân",
            f"{filename_prefix}_labels.csv",
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(frames))
        bottom = np.zeros(len(frames))
        for label in BINARY_LABELS:
            values = np.array(
                [stats[name]["binary_distribution"].get(label, 0) for name in frames]
            )
            bars = axes[0].bar(
                x, values, bottom=bottom, label=label, color=COLORS[label]
            )
            for i, bar in enumerate(bars):
                if values[i]:
                    axes[0].text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[i] + values[i] / 2,
                        f"{values[i]:,}\n({values[i] / stats[list(frames)[i]]['rows']:.1%})",
                        ha="center",
                        va="center",
                        color="white",
                    )
            bottom += values
        axes[0].set(
            xticks=x,
            xticklabels=list(frames),
            ylabel="Số mẫu",
            title="Phân bố SAFE / TOXIC",
        )
        axes[0].legend()
        max_length = max(int(frame.text.str.len().max()) for frame in frames.values())
        bins = np.linspace(0, np.log1p(max(max_length, 1)), 35)
        for name, frame in frames.items():
            lengths = frame.text.str.len().to_numpy()
            axes[1].hist(
                np.log1p(lengths),
                bins=bins,
                weights=np.ones(len(lengths)) / len(lengths),
                alpha=0.55,
                label=name,
            )
        axes[1].set(
            xlabel="log(1 + độ dài ký tự)",
            ylabel="Tỷ lệ mẫu",
            title="Phân bố toàn bộ độ dài văn bản",
        )
        axes[1].legend()
        self.show_figure(fig, f"{filename_prefix}_distributions.png")
        return stats

    def split_overview(self, train_df, dev_df):
        aggregate = {
            name: {
                "rows": len(frame),
                **{label: int((frame.label == label).sum()) for label in BINARY_LABELS},
            }
            for name, frame in {"train": train_df, "dev": dev_df}.items()
        }
        split_overview = pd.DataFrame(
            [
                {
                    "split": name,
                    **values,
                    "TOXIC ratio": values["TOXIC"] / values["rows"],
                }
                for name, values in aggregate.items()
            ]
        )
        self.show_table(
            split_overview, "Quy mô dữ liệu train/dev", "train_dev_sizes.csv"
        )

    def audit_train_dev(self, train_df, dev_df):
        audit = {
            "privacy": "aggregate_only",
            **self.display_audit({"train": train_df, "dev": dev_df}, "audit_train_dev"),
        }
        write_json(self.output_dir / "audit_train_dev.json", audit)
        return audit

    def baselines(self, results):
        ranked = results.ranked
        baseline_overview = pd.DataFrame(
            [
                {
                    "model": row["name"],
                    "train_macro_f1": row["train"]["macro"]["f1"],
                    "dev_macro_f1": row["dev"]["macro"]["f1"],
                    "train_minus_dev_f1": row["train"]["macro"]["f1"]
                    - row["dev"]["macro"]["f1"],
                    "dev_toxic_recall": row["dev"]["per_class"]["TOXIC"]["recall"],
                    "fit_seconds": row["fit_seconds"],
                    "evaluation_seconds": row["evaluation_seconds"],
                }
                for row in ranked
            ]
        )
        self.show_table(
            baseline_overview,
            "Xếp hạng baseline; thời gian tính bằng giây",
            "baseline_ranking.csv",
        )
        self.metric_tables(
            [
                (row["name"], split, row[split])
                for row in ranked
                for split in ("train", "dev")
            ],
            "baseline",
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        x = np.arange(len(ranked))
        for offset, split in [(-0.18, "train"), (0.18, "dev")]:
            bars = axes[0].bar(
                x + offset,
                baseline_overview[f"{split}_macro_f1"],
                width=0.36,
                label=split,
                color=COLORS[split],
            )
            annotate_bars(axes[0], bars, percentage=True)
        axes[0].set(
            xticks=x,
            xticklabels=baseline_overview.model,
            ylim=(0, 1.13),
            title="Macro-F1: train và dev",
        )
        axes[0].legend(ncol=2, loc="upper center", fontsize=8)
        bars = axes[1].bar(x, baseline_overview.fit_seconds, color="#579D85")
        annotate_bars(axes[1], bars, decimals=2)
        axes[1].set(
            xticks=x,
            xticklabels=baseline_overview.model,
            ylabel="Giây",
            title="Thời gian fit (gồm tiền xử lý và TF-IDF)",
        )
        axes[1].margins(y=0.18)
        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
        self.show_figure(fig, "baseline_comparison.png")
        for split in ("train", "dev"):
            self.plot_confusions(
                [(row["name"], row[split]) for row in ranked],
                f"baseline_{split}_confusions.png",
                f"Baseline • {split}",
            )
        return baseline_overview

    def selection(self, selection):
        manifest, lock = selection.manifest, selection.lock
        selected_model = lock["selected_model"]
        print(f"Đã khóa baseline trước khi mở test: {selected_model}")
        self.show_table(
            pd.json_normalize(manifest)
            .T.rename_axis("setting")
            .reset_index()
            .rename(columns={0: "value"}),
            "Manifest lần chạy",
            "manifest.csv",
        )
        self.show_table(
            pd.json_normalize(lock)
            .T.rename_axis("setting")
            .reset_index()
            .rename(columns={0: "value"}),
            "Khóa lựa chọn và checksum trước test",
            "selection_lock_before_test.csv",
        )

    def test_summary(self, test):
        lock, test_metrics, test_df = test.lock, test.metrics, test.frame
        test_predict_seconds = test.predict_seconds
        print(
            f"TEST • {lock['selected_model']} • {len(test_df):,} mẫu • dự đoán {test_predict_seconds:.2f}s"
        )
        write_json(
            self.output_dir / "audit_test.json",
            {
                "privacy": "aggregate_only",
                **self.display_audit({"test": test_df}, "audit_test"),
            },
        )
        self.metric_tables([(lock["selected_model"], "test", test_metrics)], "test")

    def test_plots(self, test):
        lock, test_metrics = test.lock, test.metrics
        self.plot_confusions(
            [(f"{lock['selected_model']} • test", test_metrics)],
            "test_confusion_matrix.png",
            "Test • baseline đã khóa",
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(2)
        for offset, metric, color in [
            (-0.25, "precision", "#2878B5"),
            (0, "recall", "#E69F00"),
            (0.25, "f1", "#579D85"),
        ]:
            bars = ax.bar(
                x + offset,
                [test_metrics["per_class"][label][metric] for label in BINARY_LABELS],
                width=0.25,
                label=metric,
                color=color,
            )
            annotate_bars(ax, bars, percentage=True)
        ax.set(
            xticks=x,
            xticklabels=BINARY_LABELS,
            ylim=(0, 1.15),
            title=f"{lock['selected_model']} • test • kết quả từng lớp",
        )
        ax.legend(loc="upper center", ncol=3, fontsize=8)
        self.show_figure(fig, "test_per_class.png")

    def errors(self, errors):
        lock = {"selected_model": errors.model_name}
        error_counts, error_rates = errors.counts, errors.rates
        length_rows, error_examples = errors.by_length, errors.examples
        private_errors = errors.payload
        fp, fn = private_errors["SAFE_as_TOXIC"], private_errors["TOXIC_as_SAFE"]
        length_labels = [row["length_chars"] for row in length_rows]
        self.show_table(
            error_counts,
            f"{errors.model_name} • {fp + fn:,}/{int(error_counts['count'].sum()):,} lỗi test",
            "test_outcomes.csv",
        )
        self.show_table(error_rates, "Tỷ lệ lỗi và mẫu số", "test_error_rates.csv")
        self.show_table(
            pd.DataFrame(length_rows),
            "Lỗi theo độ dài ký tự",
            "test_errors_by_length.csv",
        )
        self.show_table(
            pd.DataFrame(
                error_examples,
                columns=["fingerprint_prefix", "truth", "prediction", "length_chars"],
            ),
            "Tối đa 20 lỗi đầu tiên (định danh bằng fingerprint)",
            "test_error_fingerprints.csv",
        )
        write_json(self.output_dir / "error_analysis.json", private_errors)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        bars = axes[0].bar(
            ["FP: báo nhầm", "FN: bỏ lọt"], [fp, fn], color=["#E69F00", "#D9534F"]
        )
        annotate_bars(axes[0], bars)
        axes[0].set(ylabel="Số mẫu", title="Hai loại lỗi test")
        axes[0].margins(y=0.18)
        rates = [
            row["error_rate"] if row["error_rate"] is not None else np.nan
            for row in length_rows
        ]
        bars = axes[1].bar(length_labels, rates, color="#8172B3")
        annotate_bars(axes[1], bars, percentage=True)
        for index, row in enumerate(length_rows):
            if row["samples"] == 0:
                axes[1].text(index, 0.03, "N/A", ha="center", color="gray")
        axes[1].set(
            ylim=(0, 1.15),
            xlabel="Độ dài ký tự",
            ylabel="Tỷ lệ lỗi",
            title="Lỗi theo độ dài (n = số mẫu)",
        )
        axes[1].set_xticks(
            range(len(length_labels)),
            [f"{row['length_chars']}\nn={row['samples']:,}" for row in length_rows],
        )
        fig.suptitle(f"{lock['selected_model']} • phân tích hậu nghiệm trên test")
        self.show_figure(fig, "test_error_analysis.png")

    def demo(self, model_name, demo_results):
        lock = {"selected_model": model_name}
        self.show_table(
            pd.DataFrame(demo_results),
            f"Demo • {lock['selected_model']} • câu tự tạo, không phải dữ liệu test",
            "demo_predictions.csv",
        )
        write_json(
            self.output_dir / "demo_predictions.json",
            {
                "source": "synthetic_examples",
                "model": lock["selected_model"],
                "examples": demo_results,
            },
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(
            [row["example"] for row in demo_results],
            [row["confidence_proxy"] for row in demo_results],
            color=[COLORS[row["prediction"]] for row in demo_results],
        )
        annotate_bars(ax, bars, percentage=True)
        for bar, row in zip(bars, demo_results):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.03,
                row["prediction"],
                ha="center",
                color="white",
            )
        ax.axhline(0.65, linestyle="--", color="gray", label="Ngưỡng xem xét lại 0.65")
        ax.set(
            ylim=(0, 1.1),
            ylabel="Confidence proxy",
            title=f"Demo • {lock['selected_model']} • điểm chưa hiệu chỉnh",
        )
        ax.legend(loc="upper right")
        self.show_figure(fig, "demo_predictions.png")

    def neural(self, state):
        neural_history, neural_payload, RUN_ID = (
            state.history,
            state.payload,
            state.run.run_id,
        )
        if neural_history:
            self.show_table(
                pd.DataFrame(
                    [
                        {
                            "run_id": RUN_ID,
                            "device": neural_payload["device"],
                            "preprocessing_seconds": neural_payload[
                                "preprocessing_seconds"
                            ],
                            "status": neural_payload["status"],
                        }
                    ]
                ),
                "Thông tin lần chạy neural",
                "neural_run_info.csv",
            )
            neural_details = [
                {
                    "model": name,
                    **{
                        key: value
                        for key, value in neural_payload[name].items()
                        if key not in {"train", "dev"}
                    },
                }
                for name in ("bilstm", "phobert")
                if name in neural_payload
            ]
            self.show_table(
                pd.DataFrame(neural_details),
                "Checkpoint tốt nhất, cấu hình và thời gian neural",
                "neural_model_details.csv",
            )
            history_df = history_frame(neural_history)
            self.show_table(
                history_df,
                "Toàn bộ lịch sử epoch; loss/metrics train và dev đo sau mỗi epoch",
                "neural_history.csv",
            )
            epoch_entries = [
                (f"{row['model']} epoch {row['epoch']}", split, row[split])
                for row in neural_history
                for split in ("train", "dev")
            ]
            self.metric_tables(epoch_entries, "neural_epochs")
            for name in ("bilstm", "phobert"):
                if name not in neural_payload:
                    continue
                history = history_df.loc[history_df.model == name]
                best_epoch = neural_payload[name]["best_epoch"]
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for split in ("train", "dev"):
                    axes[0].plot(
                        history.epoch,
                        history[f"{split}_loss"],
                        marker="o",
                        label=f"{split} eval",
                        color=COLORS[split],
                    )
                    axes[1].plot(
                        history.epoch,
                        history[f"{split}_macro_f1"],
                        marker="o",
                        label=split,
                        color=COLORS[split],
                    )
                axes[0].plot(
                    history.epoch,
                    history.optimization_loss,
                    "--",
                    color="#999999",
                    label="train tối ưu (trung bình batch)",
                )
                axes[0].set(title="Cross-entropy loss", ylabel="Loss")
                axes[1].set(title="Macro-F1", ylabel="Điểm", ylim=(0, 1.05))
                axes[2].plot(
                    history.epoch,
                    history.dev_toxic_recall,
                    marker="o",
                    color=COLORS["TOXIC"],
                    label="dev TOXIC Recall",
                )
                axes[2].set(title="Độ phủ TOXIC trên dev", ylim=(0, 1.05))
                for ax in axes:
                    ax.axvline(
                        best_epoch,
                        color="#579D85",
                        linestyle=":",
                        label=f"Best epoch {best_epoch}",
                    )
                    ax.set_xticks(history.epoch)
                    ax.set_xlabel("Epoch")
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.2)
                fig.suptitle(f"{name} • train/dev • chọn checkpoint bằng dev")
                self.show_figure(fig, f"{name}_learning_curves.png")
                self.metric_tables(
                    [
                        (name, split, neural_payload[name][split])
                        for split in ("train", "dev")
                    ],
                    f"{name}_best",
                )
                self.plot_confusions(
                    [
                        (
                            f"{name} epoch {best_epoch} • {split}",
                            neural_payload[name][split],
                        )
                        for split in ("train", "dev")
                    ],
                    f"{name}_best_confusions.png",
                    f"{name} • checkpoint tốt nhất",
                )

    def comparison(self, baselines, neural, test):
        ranked, neural_payload = baselines.ranked, neural.payload
        lock, test_metrics = test.lock, test.metrics
        RUN_ID, RUN_MODE = self.run.run_id, self.run.config.run_mode
        comparison_rows = []
        all_dev_entries = []
        for row in ranked:
            comparison_rows.append(
                {
                    "model": row["name"],
                    "type": "ML baseline",
                    "best_epoch": None,
                    "train_macro_f1": row["train"]["macro"]["f1"],
                    "dev_accuracy": row["dev"]["accuracy"],
                    "dev_macro_f1": row["dev"]["macro"]["f1"],
                    "dev_toxic_recall": row["dev"]["per_class"]["TOXIC"]["recall"],
                    "dev_toxic_f1": row["dev"]["per_class"]["TOXIC"]["f1"],
                    "fit_seconds": row["fit_seconds"],
                    "evaluation_seconds": row["evaluation_seconds"],
                    "test_and_demo_model": row["name"] == lock["selected_model"],
                }
            )
            all_dev_entries.append((row["name"], "dev", row["dev"]))
        for name in ("bilstm", "phobert"):
            if name not in neural_payload:
                continue
            row = neural_payload[name]
            comparison_rows.append(
                {
                    "model": name,
                    "type": "Neural",
                    "best_epoch": row["best_epoch"],
                    "train_macro_f1": row["train"]["macro"]["f1"],
                    "dev_accuracy": row["dev"]["accuracy"],
                    "dev_macro_f1": row["dev"]["macro"]["f1"],
                    "dev_toxic_recall": row["dev"]["per_class"]["TOXIC"]["recall"],
                    "dev_toxic_f1": row["dev"]["per_class"]["TOXIC"]["f1"],
                    "fit_seconds": row["fit_seconds"],
                    "evaluation_seconds": row["evaluation_seconds"],
                    "test_and_demo_model": False,
                }
            )
            all_dev_entries.append((name, "dev", row["dev"]))
        leaderboard = (
            pd.DataFrame(comparison_rows)
            .sort_values(
                ["dev_macro_f1", "dev_toxic_recall", "model"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
        leaderboard["train_minus_dev_f1"] = (
            leaderboard.train_macro_f1 - leaderboard.dev_macro_f1
        )
        self.show_table(
            leaderboard,
            "Bảng tổng hợp trên DEV (giá trị lưu không bị làm tròn)",
            "all_models_dev_leaderboard.csv",
        )
        self.metric_tables(all_dev_entries, "all_models_dev")
        write_json(
            self.output_dir / "all_models_dev_leaderboard.json",
            {
                "run_id": RUN_ID,
                "split": "dev",
                "models": json.loads(
                    leaderboard.to_json(orient="records", double_precision=15)
                ),
                "test_and_demo_model": lock["selected_model"],
            },
        )
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x = np.arange(len(leaderboard))
        for offset, column, label, color in [
            (-0.18, "dev_macro_f1", "Dev Macro-F1", "#2878B5"),
            (0.18, "dev_toxic_recall", "Dev TOXIC Recall", "#E69F00"),
        ]:
            bars = axes[0, 0].bar(
                x + offset, leaderboard[column], width=0.36, label=label, color=color
            )
            annotate_bars(axes[0, 0], bars, percentage=True)
        axes[0, 0].set(title="Chất lượng trên dev", ylim=(0, 1.18))
        axes[0, 0].legend(fontsize=8, ncol=2, loc="upper center")
        for offset, split in [(-0.18, "train"), (0.18, "dev")]:
            bars = axes[0, 1].bar(
                x + offset,
                leaderboard[f"{split}_macro_f1"],
                width=0.36,
                label=split,
                color=COLORS[split],
            )
            annotate_bars(axes[0, 1], bars, percentage=True)
        axes[0, 1].set(title="Macro-F1: train và dev", ylim=(0, 1.18))
        axes[0, 1].legend(fontsize=8, ncol=2, loc="upper center")
        bars = axes[1, 0].bar(x, leaderboard.fit_seconds, color="#579D85")
        annotate_bars(axes[1, 0], bars, decimals=2)
        axes[1, 0].set(
            title="Thời gian fit / tối ưu (xem phạm vi ở trên)", ylabel="Giây"
        )
        axes[1, 0].margins(y=0.18)
        heat_columns = [
            "dev_accuracy",
            "dev_macro_f1",
            "dev_toxic_recall",
            "dev_toxic_f1",
        ]
        heat_values = leaderboard[heat_columns].to_numpy()
        axes[1, 1].imshow(heat_values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axes[1, 1].set(
            xticks=range(4),
            xticklabels=["Accuracy", "Macro-F1", "TOXIC Recall", "TOXIC F1"],
            yticks=x,
            yticklabels=leaderboard.model,
            title="Dev metrics (0–1)",
        )
        for i in range(len(leaderboard)):
            for j in range(4):
                axes[1, 1].text(
                    j,
                    i,
                    f"{heat_values[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color="white" if heat_values[i, j] > 0.5 else "black",
                )
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xticks(x, leaderboard.model, rotation=20, ha="right")
        self.show_figure(fig, "all_models_comparison.png")
        print(
            f"Model đứng đầu DEV: {leaderboard.iloc[0]['model']} (Macro-F1={leaderboard.iloc[0]['dev_macro_f1']:.4f})"
        )
        print(
            f"Model TEST và DEMO đã khóa: {lock['selected_model']} "
            f"(test Accuracy={test_metrics['accuracy']:.4f}; test Macro-F1={test_metrics['macro']['f1']:.4f})"
        )
        if RUN_MODE == "SMOKE":
            print(
                "SMOKE: kết quả chỉ dùng kiểm tra luồng, không đưa vào báo cáo chất lượng mô hình."
            )
        return leaderboard

    def export(self):
        artifact_rows = []
        for file_path in sorted(self.output_dir.rglob("*")):
            if file_path.is_file():
                artifact_rows.append(
                    {
                        "file": str(file_path.relative_to(self.output_dir)),
                        "bytes": file_path.stat().st_size,
                        "size_MiB": file_path.stat().st_size / (1024**2),
                        "type": file_path.suffix,
                        "absolute_path": str(file_path.resolve()),
                    }
                )
        self.show_table(
            pd.DataFrame(artifact_rows),
            "Các artifacts đã tạo (không liệt kê chính bảng inventory và ZIP tạo sau)",
            "artifact_inventory.csv",
        )
        report_archive = self.output_dir / "reports.zip"
        with zipfile.ZipFile(
            report_archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for file_path in sorted(self.output_dir.iterdir()):
                if file_path.is_file() and file_path.suffix in {
                    ".json",
                    ".csv",
                    ".png",
                }:
                    archive.write(file_path, arcname=file_path.name)
        print(f"Đã lưu {len(artifact_rows)} artifacts trước inventory/ZIP.")
        print(f"Thư mục: {self.output_dir.resolve()}")
        print(
            f"ZIP báo cáo: {report_archive.resolve()} ({report_archive.stat().st_size / 1024**2:.2f} MiB)"
        )
        print(
            "Colab: mở bảng Files bên trái → thư mục lần chạy → reports.zip → Download."
        )
        return report_archive

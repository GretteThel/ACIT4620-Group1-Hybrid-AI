"""Validate saved validation predictions and export comparable CSVs and figures."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ham10000_hybrid.metrics import compute_classification_metrics

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall")
TITLES = ("Accuracy", "Balanced accuracy", "Macro-F1", "Melanoma recall")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_saved_run(folder, label=None):
    folder = folder.resolve()
    status = read_json(folder / "run_status.json")
    if status["state"] != "completed":
        raise ValueError(f"Run has not completed: {folder}")
    config = read_json(folder / "experiment_config.json")
    if config["test_evaluated"]:
        raise ValueError("This comparison is restricted to validation-only experiments")
    codes = config["class_codes"]
    metrics = read_json(folder / "validation_metrics.json")
    with np.load(folder / "validation_predictions.npz", allow_pickle=False) as saved:
        probs, truth, ids = saved["probs"], saved["y_true"], saved["image_id"]
        if probs.shape != (len(truth), len(codes)) or not len(truth):
            raise ValueError("Prediction dimensions do not match labels/classes")
        if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Invalid validation probabilities")
        if not np.issubdtype(truth.dtype, np.integer) or (truth < 0).any() or (truth >= len(codes)).any():
            raise ValueError("Invalid ground-truth indices")
        if len(ids) != len(truth) or len(set(ids)) != len(ids):
            raise ValueError("Validation image IDs must be unique and match predictions")
        code_array = np.asarray(codes)
        computed = compute_classification_metrics(
            code_array[truth].tolist(), code_array[probs.argmax(axis=1)].tolist(),
        )
        identity = pd.DataFrame({"image_id": ids, "label": truth}).sort_values("image_id")
        identity = identity.reset_index(drop=True)
    for key in METRICS:
        if not np.isclose(metrics[key], computed[key], atol=1e-8):
            raise ValueError(f"Saved metric differs from predictions: {key}")
    manifest = pd.read_csv(folder / "split_manifest_summary.csv")
    expected = manifest.loc[manifest["split"] == "validation", ["image_id", "label"]]
    expected = expected.sort_values("image_id").reset_index(drop=True)
    if not identity.equals(expected):
        raise ValueError("Saved validation predictions differ from the manifest")
    history = pd.read_csv(folder / "training_history.csv")
    required = {"stage", "epoch", "train_loss", "val_loss", *[f"val_{key}" for key in METRICS]}
    if not required.issubset(history.columns) or history.empty:
        raise ValueError("Missing or empty epoch history")
    history["run_id"] = folder.name
    history["run_label"] = label or folder.name
    history["local_epoch"] = np.arange(1, len(history) + 1)
    warm = config.get("warm_start", {})
    offset = int(warm.get("cumulative_epochs_at_initialization", 0))
    history["checkpoint_chain_epoch"] = history["local_epoch"] + offset
    checkpoint = folder / "best_efficientnet_b0.pth"
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "folder": folder, "run_id": folder.name, "label": label or folder.name,
        "config": config, "metrics": metrics, "computed": computed, "identity": identity,
        "history": history, "checkpoint_sha256": checkpoint_sha,
    }


def validate_comparison(runs):
    reference = runs[0]
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("Use distinct run folder names")
    for run in runs[1:]:
        for key in ("split_sha256", "class_codes"):
            if run["config"][key] != reference["config"][key]:
                raise ValueError(f"Cannot compare runs with different {key}")
        if not run["identity"].equals(reference["identity"]):
            raise ValueError("Runs use different validation image IDs or labels")


def export_comparison(runs, out):
    validate_comparison(runs)
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [{
        "run_id": run["run_id"], "run_label": run["label"],
        "completed_local_epochs": len(run["history"]),
        "warm_start": bool(run["config"].get("warm_start")),
        "class_weights": run["config"]["use_class_weights"],
        **{key: run["metrics"][key] for key in METRICS},
    } for run in runs]
    table = pd.DataFrame(rows)
    table.to_csv(out / "run_metrics.csv", index=False)
    pd.concat([run["history"] for run in runs], ignore_index=True).to_csv(
        out / "epoch_history.csv", index=False,
    )
    class_table = pd.DataFrame([
        {"run_id": run["run_id"], "run_label": run["label"], "class": code,
         **run["computed"]["per_class"][code]}
        for run in runs for code in run["config"]["class_codes"]
    ])
    class_table.to_csv(out / "class_metrics.csv", index=False)
    colors = plt.get_cmap("tab10").colors

    def save_figure(fig, stem):
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    for ax, key, title in zip(axes, METRICS, TITLES):
        values = table[key].to_numpy()
        bars = ax.bar(np.arange(len(runs)), values, color=colors[:len(runs)])
        ax.set_xticks(np.arange(len(runs)), [run["label"].replace(" ", "\n") for run in runs])
        ax.set(title=title, ylim=(0, 1), ylabel="Validation score")
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "01_metric_comparison")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = [("val_macro_f1", "Macro-F1"), ("val_balanced_accuracy", "Balanced accuracy"),
             ("val_melanoma_recall", "Melanoma recall"), ("val_loss", "Validation loss")]
    for ax, (key, title) in zip(axes.flat, specs):
        for index, run in enumerate(runs):
            history = run["history"]
            ax.plot(history["local_epoch"], history[key], marker="o", markersize=3,
                    label=run["label"], color=colors[index % len(colors)])
        ax.set(title=title, xlabel="Epoch within each run (starting weights differ)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save_figure(fig, "02_learning_curves_by_run")

    codes = runs[0]["config"]["class_codes"]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.8 / len(runs)
    for index, run in enumerate(runs):
        values = [run["computed"]["per_class"][code]["recall"] for code in codes]
        x = np.arange(len(codes)) - 0.4 + width / 2 + index * width
        ax.bar(x, values, width=width, label=run["label"], color=colors[index % len(colors)])
    ax.set_xticks(np.arange(len(codes)), codes)
    ax.set(title="Validation recall by class", ylabel="Recall", ylim=(0, 1))
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "03_class_recall_comparison")

    by_path = {run["folder"]: run for run in runs}
    chains = [(run, by_path.get(Path(run["config"]["warm_start"]["source_run"]).resolve()))
              for run in runs if run["config"].get("warm_start")]
    chains = [(child, parent) for child, parent in chains if parent is not None]
    if chains:
        fig, axes = plt.subplots(len(chains), 2, figsize=(13, 4.8 * len(chains)), squeeze=False)
        for (child, parent), ax_row in zip(chains, axes):
            boundary = child["config"]["warm_start"]["cumulative_epochs_at_initialization"]
            for ax, key, title in zip(ax_row, ["val_macro_f1", "val_melanoma_recall"],
                                      ["Macro-F1", "Melanoma recall"]):
                for run in [parent, child]:
                    h = run["history"]
                    if run is parent:
                        h = h[h["checkpoint_chain_epoch"] <= boundary]
                    ax.plot(h["checkpoint_chain_epoch"], h[key], marker="o", label=run["label"])
                ax.axvline(boundary + 0.5, color="gray", linestyle="--", label="Warm start; new optimizer")
                ax.set(title=title, xlabel="Epoch along the checkpoint lineage")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        save_figure(fig, "04_checkpoint_lineage_curves")

    metadata = {
        "evaluation_split": "validation", "split_sha256": runs[0]["config"]["split_sha256"],
        "class_codes": codes, "validation_images": len(runs[0]["identity"]),
        "verification": "Saved probabilities, IDs, labels and recomputed metrics matched",
        "runs": [{"folder": str(run["folder"]), "label": run["label"],
                  "checkpoint_sha256": run["checkpoint_sha256"], "config": run["config"]}
                 for run in runs],
    }
    (out / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    header = "| Run | Epochs in this run | Accuracy | Balanced accuracy | Macro-F1 | Melanoma recall |"
    report_rows = [
        f"| {run['label']} | {len(run['history'])} | {run['metrics']['accuracy']:.2%} | "
        f"{run['metrics']['balanced_accuracy']:.2%} | {run['metrics']['macro_f1']:.4f} | "
        f"{run['metrics']['melanoma_recall']:.2%} |" for run in runs
    ]
    report = "# CNN 실행 비교\n\n" + header + "\n|---|---:|---:|---:|---:|---:|\n"
    report += "\n".join(report_rows)
    report += "\n\n같은 검증 이미지와 분할에서 얻은 단일 seed 결과입니다. 최종 테스트 성능이 아닙니다.\n"
    report += "추가 미세조정은 부모 가중치에서 시작하며 optimizer와 RNG 상태는 새로 초기화합니다.\n"
    report += "각 지표는 마지막 epoch가 아니라 해당 실행에서 선택한 최적 체크포인트의 결과입니다.\n"
    report += "\nCSV의 점수는 0–1 단위 원본 값입니다. 이후 그래프를 다시 그릴 때 사용할 수 있습니다.\n\n"
    for filename, title in [
        ("run_metrics.csv", "실행별 지표 CSV"), ("epoch_history.csv", "Epoch별 원본 이력 CSV"),
        ("class_metrics.csv", "클래스별 지표 CSV"),
        ("01_metric_comparison.png", "실행별 지표 비교"),
        ("02_learning_curves_by_run.png", "실행별 학습 곡선"),
        ("03_class_recall_comparison.png", "클래스별 재현율 비교"),
        ("04_checkpoint_lineage_curves.png", "추가 미세조정의 누적 학습 곡선"),
    ]:
        if (out / filename).is_file():
            report += f"- [{title}]({filename})\n"
    report += "\n모든 그래프는 편집 가능한 SVG도 함께 저장했습니다. 실행별 원본 폴더는 comparison_metadata.json에 있습니다.\n"
    (out / "RESULTS.md").write_text(report, encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.runs):
        parser.error("Supply one label for each run")
    labels = args.labels or [path.name for path in args.runs]
    runs = [load_saved_run(path, label) for path, label in zip(args.runs, labels)]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.output_dir or ROOT / "reports/comparisons" / stamp
    print(export_comparison(runs, out).to_string(index=False))
    print("Comparison saved:", out.resolve())


if __name__ == "__main__":
    main()

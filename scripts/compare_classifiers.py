"""Compare real final class predictions with paired lesion difference intervals."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from compare_runs import load_saved_run, validate_comparison
from evaluate_review import load_bundle

from ham10000_hybrid.classifier_comparison import (
    METRIC_NAMES,
    align_hybrid_predictions,
    paired_lesion_comparison,
)
from ham10000_hybrid.data_validation import CLASS_CODES
from ham10000_hybrid.evaluation import probability_metrics
from ham10000_hybrid.metrics import compute_classification_metrics

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def evaluate(args):
    baseline = load_saved_run(args.baseline_run)
    data = load_bundle(baseline)
    y, baseline_probs, ids = data["y"], data["probs"], data["ids"]
    metadata = {"baseline_run": str(baseline["folder"]),
                "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
                "split_sha256": baseline["config"]["split_sha256"],
                "evaluation_split": "validation", "test_evaluated": False}
    controls = {}
    if args.candidate_run:
        candidate = load_saved_run(args.candidate_run)
        validate_comparison([baseline, candidate])
        alternative = load_bundle(candidate)
        positions = {image_id: i for i, image_id in enumerate(alternative["ids"])}
        order = np.array([positions[image_id] for image_id in ids])
        if (not np.array_equal(y, alternative["y"][order])
                or not np.array_equal(data["lesions"], alternative["lesions"][order])):
            raise ValueError("Candidate labels/lesions differ from the source validation manifest")
        candidate_probs = alternative["probs"][order]
        candidate_pred = candidate_probs.argmax(axis=1)
        metadata.update({"candidate_kind": "cnn_training_variant",
                         "candidate_run": str(candidate["folder"]),
                         "candidate_checkpoint_sha256": candidate["checkpoint_sha256"]})
        for key in ["seed", "head_epochs", "finetune_epochs", "batch_size", "pretraining",
                    "head_lr", "backbone_lr", "fine_tune_head_lr", "class_codes"]:
            controls[key] = {"baseline": baseline["config"].get(key),
                             "candidate": candidate["config"].get(key),
                             "matches": baseline["config"].get(key) == candidate["config"].get(key)}
        label = "Candidate CNN"
    else:
        frame = pd.read_csv(args.hybrid_predictions, dtype={"image_id": str})
        producer = json.loads(args.hybrid_metadata.read_text(encoding="utf-8"))
        aligned = align_hybrid_predictions(
            frame, producer, ids, source_run_id=baseline["run_id"],
            source_checkpoint_sha256=baseline["checkpoint_sha256"],
            split_sha256=baseline["config"]["split_sha256"],
        )
        candidate_probs, candidate_pred = aligned["probs"], aligned["predicted"]
        metadata.update({"candidate_kind": "neuro_fuzzy_classifier", "producer_metadata": producer,
                         "hybrid_predictions_sha256": hashlib.sha256(args.hybrid_predictions.read_bytes()).hexdigest(),
                         "hybrid_metadata_sha256": hashlib.sha256(args.hybrid_metadata.read_bytes()).hexdigest(),
                         "producer_fit_split_audit": "Declaration checked; actual fitting procedure not independently verified"})
        label = "CNN + neuro-fuzzy"
    metadata["controlled_settings"] = controls
    baseline_pred = baseline_probs.argmax(axis=1)
    comparison = paired_lesion_comparison(y, baseline_pred, candidate_pred, data["lesions"],
                                         resamples=args.bootstrap_resamples, seed=args.seed)
    reports = {}
    for name, pred, probabilities in [("CNN baseline", baseline_pred, baseline_probs),
                                      (label, candidate_pred, candidate_probs)]:
        report = compute_classification_metrics([CLASS_CODES[i] for i in y],
                                                [CLASS_CODES[i] for i in pred])
        report["probability_metrics"] = probability_metrics(y, probabilities) if probabilities is not None else None
        reports[name] = report
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "paired_comparison.json", comparison)
    write_json(out / "classification_reports.json", reports)
    write_json(out / "comparison_metadata.json", metadata)
    delta = pd.DataFrame([{"metric": key, **comparison["metrics"][key]} for key in METRIC_NAMES])
    delta.to_csv(out / "metric_differences.csv", index=False)
    pd.DataFrame({"image_id": ids, "lesion_id": data["lesions"],
                  "true_dx": [CLASS_CODES[i] for i in y],
                  "baseline_dx": [CLASS_CODES[i] for i in baseline_pred],
                  "candidate_dx": [CLASS_CODES[i] for i in candidate_pred],
                  "baseline_correct": baseline_pred == y, "candidate_correct": candidate_pred == y}).to_csv(
        out / "paired_image_predictions.csv", index=False,
    )
    pd.DataFrame([{"classifier": name, "class": code, **report["per_class"][code]}
                  for name, report in reports.items() for code in CLASS_CODES]).to_csv(
        out / "class_metrics.csv", index=False,
    )
    if not args.no_plots:
        plot_comparison(out, comparison, reports)
    write_report(out, comparison, metadata, label)
    print(delta.to_string(index=False))
    print("Error transitions:", comparison["transitions"])
    print("Saved classifier comparison:", out)


def plot_comparison(out, comparison, reports):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/.matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        for suffix in ["png", "svg"]:
            fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(METRIC_NAMES))
    for index, (name, report) in enumerate(reports.items()):
        axes[0].bar(x + (index - .5) * .35, [report[key] for key in METRIC_NAMES], .35, label=name)
    axes[0].set_xticks(x, ["Accuracy", "Balanced\naccuracy", "Macro-F1", "Melanoma\nrecall"])
    axes[0].set(title="Validation classification metrics", ylim=(0, 1))
    axes[0].legend()
    axes[0].grid(axis="y", alpha=.2)
    for i, key in enumerate(METRIC_NAMES):
        values = comparison["metrics"][key]
        if values["lower_95"] is not None:
            axes[1].vlines(i, values["lower_95"], values["upper_95"], color="tab:blue", linewidth=3)
            axes[1].scatter(i, values["difference"], color="black", zorder=3)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set_xticks(x, ["Accuracy", "Balanced\naccuracy", "Macro-F1", "Melanoma\nrecall"])
    axes[1].set(title="Paired lesion bootstrap: candidate - baseline", ylabel="Score difference; 95% percentile interval")
    axes[1].grid(axis="y", alpha=.2)
    save(fig, "01_classification_and_paired_differences")
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, (name, report) in enumerate(reports.items()):
        ax.bar(np.arange(7) + (index - .5) * .35,
               [report["per_class"][code]["recall"] for code in CLASS_CODES], .35, label=name)
    ax.set_xticks(np.arange(7), CLASS_CODES)
    ax.set(title="Final class recall", ylim=(0, 1), ylabel="Recall")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    save(fig, "02_class_recall")


def write_report(out, comparison, metadata, label):
    rows = []
    for key, item in comparison["metrics"].items():
        def fmt(value):
            return f"{value:.4f}" if value is not None else "n/a"
        rows.append(f"| {key} | {fmt(item['baseline'])} | {fmt(item['candidate'])} | "
                    f"{fmt(item['difference'])} | [{fmt(item['lower_95'])}, {fmt(item['upper_95'])}] |")
    text = "# 최종 클래스 예측의 비교평가\n\n"
    text += f"평가 대상: **{label}**, 종류: `{metadata['candidate_kind']}`. 검증셋 평가입니다.\n\n"
    text += "| 지표 | CNN 기준 | 비교 대상 | 차이 | 차이의 95% 구간 |\n|---|---:|---:|---:|---|\n"
    text += "\n".join(rows)
    transition = comparison["transitions"]
    text += (f"\n\n기존 오분류 {transition['corrected_errors']}건을 교정했고, "
             f"새 오분류 {transition['introduced_errors']}건이 생겼습니다. "
             f"흑색종 오분류는 {transition['corrected_melanoma_misses']}건 교정, "
             f"{transition['introduced_melanoma_misses']}건 새로 발생했습니다.\n")
    text += "\n같은 병변을 두 예측에 함께 재표집해 차이 구간을 계산했습니다. "
    text += "이 구간은 고정된 모델과 검증셋에 조건부이며, 학습 seed 변동이나 최종 테스트 성능을 보장하지 않습니다.\n"
    if metadata["candidate_kind"] == "cnn_training_variant":
        text += "\n이번 비교는 CNN 학습 방식의 비교이며, 뉴로퍼지 성능 향상 증거는 아닙니다.\n"
    else:
        text += "\n실제 최종 클래스 예측을 평가했으며, 검토 표시를 정답으로 간주하지 않았습니다. "
        text += "학습 분할 선언은 확인하지만 실제 뉴로퍼지 학습 과정을 독립 감사한 것은 아닙니다.\n"
    text += "\n- [지표 차이 CSV](metric_differences.csv)\n- [짝지은 이미지별 예측](paired_image_predictions.csv)\n"
    text += "- [클래스별 지표](class_metrics.csv)\n- [전체 분류 보고서](classification_reports.json)\n"
    text += "- [재표집·교정/악화 건수](paired_comparison.json)\n- [모델 출처·비교 설정](comparison_metadata.json)\n"
    for filename in ["01_classification_and_paired_differences.png", "02_class_recall.png"]:
        if (out / filename).exists():
            text += f"- [그래프]({filename})\n"
    (out / "RESULTS.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--candidate-run", type=Path)
    choice.add_argument("--hybrid-predictions", type=Path)
    parser.add_argument("--hybrid-metadata", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if bool(args.hybrid_predictions) != bool(args.hybrid_metadata):
        parser.error("Hybrid predictions and metadata must be supplied together")
    if args.bootstrap_resamples < 2:
        parser.error("At least two bootstrap resamples are required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    args.output_dir = (args.output_dir or ROOT / "reports/classifier_comparisons" / stamp).resolve()
    evaluate(args)


if __name__ == "__main__":
    main()

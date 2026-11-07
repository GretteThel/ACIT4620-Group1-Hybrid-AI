"""Evaluate saved validation predictions and matched-workload review strategies."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from compare_runs import load_saved_run, validate_comparison
from sklearn.metrics import precision_recall_curve, roc_curve

from ham10000_hybrid.data_validation import CLASS_CODES
from ham10000_hybrid.evaluation import (
    MEL_INDEX,
    align_review_scores,
    lesion_bootstrap_intervals,
    priority_order,
    probability_metrics,
    review_priority_scores,
    review_workload_curves,
    validate_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_bundle(run):
    if run["config"]["class_codes"] != list(CLASS_CODES):
        raise ValueError("Evaluation requires the fixed project class order")
    with np.load(run["folder"] / "validation_predictions.npz", allow_pickle=False) as saved:
        y, probs, ids = saved["y_true"], saved["probs"], saved["image_id"]
        tta = saved["tta_probs"] if "tta_probs" in saved.files else None
    validate_predictions(y, probs, ids, tta)
    # Some earlier NPZs store lesion IDs as object arrays. Never enable pickle;
    # recover these identifiers from the independently verified CSV manifest.
    manifest = pd.read_csv(run["folder"] / "split_manifest_summary.csv")
    if "lesion_id" not in manifest:
        raise ValueError("Manifest must include lesion_id for grouped evaluation")
    rows = manifest.set_index("image_id").loc[ids]
    if not rows["split"].eq("validation").all() or not np.array_equal(rows["label"], y):
        raise ValueError("Predictions do not match validation manifest rows")
    if rows["lesion_id"].isna().any():
        raise ValueError("Missing lesion ID")
    return {"y": y, "probs": probs, "ids": ids, "tta": tta,
            "lesions": rows["lesion_id"].astype(str).to_numpy()}


def melanoma_threshold_table(y, probs):
    """Binary melanoma-flag sweep only; does not replace seven-class predictions."""
    scores = probs[:, MEL_INDEX]
    positive = y == MEL_INDEX
    rows = []
    for threshold in np.unique(np.r_[0., scores, 1.]):
        flag = scores >= threshold
        tp, fp = int((flag & positive).sum()), int((flag & ~positive).sum())
        fn, tn = int((~flag & positive).sum()), int((~flag & ~positive).sum())
        rows.append({"threshold": threshold, "true_positive": tp, "false_positive": fp,
                     "false_negative": fn, "true_negative": tn, "n_flagged": int(flag.sum()),
                     "flag_fraction": float(flag.mean()),
                     "precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / (tp + fn) if tp + fn else None,
                     "specificity": tn / (tn + fp) if tn + fp else None})
    return pd.DataFrame(rows)


def evaluate(args):
    runs = [load_saved_run(path) for path in args.runs]
    validate_comparison(runs)
    bundles = {run["run_id"]: load_bundle(run) for run in runs}
    fuzzy = pd.read_csv(args.fuzzy_scores, dtype={"run_id": str, "image_id": str}) if args.fuzzy_scores else None
    if fuzzy is not None:
        required = {"run_id", "image_id", "review_score", "checkpoint_sha256", "split_sha256"}
        if not required.issubset(fuzzy.columns):
            raise ValueError(f"Fuzzy CSV missing columns: {sorted(required - set(fuzzy.columns))}")
        if fuzzy.empty or fuzzy["run_id"].isna().any():
            raise ValueError("Fuzzy CSV must have nonmissing run IDs and score rows")
        unknown = set(fuzzy["run_id"]) - set(bundles)
        if unknown:
            raise ValueError(f"Unknown run IDs in fuzzy CSV: {sorted(unknown)}")
    prepared = []
    for run in runs:
        data = bundles[run["run_id"]]
        strategies = review_priority_scores(data["probs"], data["tta"])
        if fuzzy is not None and run["run_id"] in set(fuzzy["run_id"]):
            selected = fuzzy.loc[fuzzy["run_id"] == run["run_id"]]
            for key, expected in [("checkpoint_sha256", run["checkpoint_sha256"]),
                                  ("split_sha256", run["config"]["split_sha256"])]:
                if not selected[key].eq(expected).all():
                    raise ValueError(f"Fuzzy scores use a different {key}")
            strategies["fuzzy"] = align_review_scores(
                selected, data["ids"],
            )
        prepared.append((run, data, strategies))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    classification_rows, probability_rows, all_curves, score_rows = [], [], [], []
    pr_curves, roc_curves, threshold_tables, details = [], [], [], {}
    for run, data, strategies in prepared:
        run_id = run["run_id"]
        y, probs, ids = data["y"], data["probs"], data["ids"]
        predicted = probs.argmax(axis=1)
        probability = probability_metrics(y, probs)
        detail = {"classification": run["computed"], "probability": probability}
        if args.bootstrap_resamples:
            detail["bootstrap"] = lesion_bootstrap_intervals(
                y, probs, data["lesions"], resamples=args.bootstrap_resamples, seed=args.seed,
            )
        details[run_id] = detail
        classification_rows.append({
            "run_id": run_id, **{key: run["metrics"][key]
                                     for key in ["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall"]},
            "melanoma_precision": run["computed"]["per_class"]["mel"]["precision"],
            "melanoma_f1": run["computed"]["per_class"]["mel"]["f1-score"],
            "melanoma_support": int((y == MEL_INDEX).sum()),
            "melanoma_correct": int(((y == MEL_INDEX) & (predicted == MEL_INDEX)).sum()),
            "melanoma_roc_auc": probability["per_class"]["mel"]["roc_auc"],
            "melanoma_average_precision": probability["per_class"]["mel"]["average_precision"],
            "nll": probability["nll"], "brier_sum": probability["multiclass_brier_sum"],
            "top_label_ece": probability["top_label_ece"],
        })
        probability_rows.extend({"run_id": run_id, "class": code, **metrics}
                                for code, metrics in probability["per_class"].items())
        curves = review_workload_curves(y, probs, ids, strategies, args.budgets)
        curves.insert(0, "run_id", run_id)
        all_curves.append(curves)
        for method, values in strategies.items():
            order = priority_order(values, ids)
            ranks = np.empty(len(y), dtype=int)
            ranks[order] = np.arange(1, len(y) + 1)
            score_rows.extend({
                "run_id": run_id, "image_id": str(ids[i]), "lesion_id": str(data["lesions"][i]),
                "true_dx": CLASS_CODES[y[i]], "predicted_dx": CLASS_CODES[predicted[i]],
                "strategy": method, "review_score": float(values[i]), "priority_rank": int(ranks[i]),
            } for i in range(len(y)))
        threshold = melanoma_threshold_table(y, probs)
        threshold.insert(0, "run_id", run_id)
        threshold_tables.append(threshold)
        positive = y == MEL_INDEX
        if positive.any():
            precision, recall, thresholds = precision_recall_curve(positive, probs[:, MEL_INDEX])
            pr_curves.append(pd.DataFrame({"run_id": run_id, "precision": precision, "recall": recall,
                                           "threshold": np.r_[thresholds, np.nan]}))
        if len(np.unique(positive)) == 2:
            fpr, tpr, thresholds = roc_curve(positive, probs[:, MEL_INDEX])
            roc_curves.append(pd.DataFrame({"run_id": run_id, "false_positive_rate": fpr,
                                            "true_positive_rate": tpr,
                                            "threshold": np.where(np.isfinite(thresholds), thresholds, np.nan)}))
        print("Evaluated:", run_id, "strategies:", ", ".join(strategies), flush=True)
    classification = pd.DataFrame(classification_rows)
    workloads = pd.concat(all_curves, ignore_index=True)
    classification.to_csv(args.output_dir / "classification_metrics.csv", index=False)
    pd.DataFrame(probability_rows).to_csv(args.output_dir / "class_probability_metrics.csv", index=False)
    workloads.to_csv(args.output_dir / "review_workload_curves.csv", index=False)
    pd.DataFrame(score_rows).to_csv(args.output_dir / "review_priority_scores.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(args.output_dir / "melanoma_threshold_sweep.csv", index=False)
    if pr_curves:
        pd.concat(pr_curves, ignore_index=True).to_csv(args.output_dir / "melanoma_pr_curves.csv", index=False)
    if roc_curves:
        pd.concat(roc_curves, ignore_index=True).to_csv(args.output_dir / "melanoma_roc_curves.csv", index=False)
    write_json(args.output_dir / "evaluation_details.json", details)
    metadata = {
        "evaluation_split": "validation", "test_evaluated": False,
        "class_codes": list(CLASS_CODES), "split_sha256": runs[0]["config"]["split_sha256"],
        "budgets": args.budgets, "review_count_policy": "floor(N * budget)",
        "tie_breaking": "Higher score first, then ascending image_id; no labels used",
        "fuzzy_evaluated_for": [run["run_id"] for run, _, scores in prepared if "fuzzy" in scores],
        "fuzzy_status": "supplied scores evaluated" if fuzzy is not None else "pending; no fuzzy scores supplied",
        "bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed,
        "probability_note": "Saved probabilities may already be temperature-fitted on this validation set",
        "runs": [{"run_id": run["run_id"], "folder": str(run["folder"]), "config": run["config"]}
                 for run in runs],
        "fuzzy_csv_sha256": hashlib.sha256(args.fuzzy_scores.read_bytes()).hexdigest() if args.fuzzy_scores else None,
    }
    write_json(args.output_dir / "evaluation_metadata.json", metadata)
    make_plots(args.output_dir, classification, workloads, pr_curves, details)
    write_report(args.output_dir, classification, workloads, metadata)
    print(classification.to_string(index=False))
    print("Saved:", args.output_dir)


def make_plots(out, classification, workloads, pr_curves, details):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/.matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {run: f"Run {i + 1}" for i, run in enumerate(classification["run_id"])}

    def save(fig, name):
        fig.tight_layout()
        for extension in ["png", "svg"]:
            fig.savefig(out / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for curve in pr_curves:
        run_id = curve["run_id"].iloc[0]
        ap = details[run_id]["probability"]["per_class"]["mel"]["average_precision"]
        ax.plot(curve["recall"], curve["precision"], label=f"{labels[run_id]} (AP={ap:.3f})" if ap is not None else labels[run_id])
    prevalence = next(iter(details.values()))["probability"]["per_class"]["mel"]["prevalence"]
    ax.axhline(prevalence, color="gray", linestyle="--", label="Melanoma prevalence")
    ax.set(title="Melanoma vs rest: validation precision-recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "01_melanoma_precision_recall")

    fig, axes = plt.subplots(len(labels), 3, figsize=(16, 4.2 * len(labels)), squeeze=False)
    for (run_id, label), row_axes in zip(labels.items(), axes):
        subset = workloads[workloads["run_id"] == run_id]
        for ax, metric, title in zip(row_axes,
                ["error_capture_rate", "missed_melanoma_capture_rate", "n_missed_melanoma_unreviewed"],
                ["CNN errors flagged", "Missed melanoma flagged", "Missed melanoma left unreviewed"]):
            for strategy, rows in subset.groupby("strategy", sort=False):
                ax.plot(rows["review_fraction"], rows[metric], marker="o", label=strategy)
            ax.set(title=f"{label}: {title}", xlabel="Fraction of images reviewed")
            if metric.endswith("rate"):
                ax.set_ylim(0, 1)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    save(fig, "02_review_workload_comparison")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect top-label calibration")
    for run_id, label in labels.items():
        bins = [value for value in details[run_id]["probability"]["ece_bins"] if value["count"]]
        ax.plot([value["mean_confidence"] for value in bins], [value["accuracy"] for value in bins],
                marker="o", label=label)
    ax.set(title="Validation top-label reliability (validation reused for calibration)",
           xlabel="Mean top-label confidence", ylabel="Observed argmax accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "03_top_label_reliability")


def write_report(out, classification, workloads, metadata):
    lines = ["# CNN·검토 우선순위 비교평가", "", "검증셋 평가입니다. 최종 테스트셋은 사용하지 않았습니다.", ""]
    for index, row in classification.iterrows():
        lines.append(f"- Run {index + 1}: `{row['run_id']}`")
    lines += ["", "| Run | Macro-F1 | Melanoma recall | Melanoma AP | Melanoma ROC-AUC |",
              "|---|---:|---:|---:|---:|"]
    for index, row in classification.iterrows():
        ap = f"{row['melanoma_average_precision']:.4f}" if pd.notna(row["melanoma_average_precision"]) else "n/a"
        auc = f"{row['melanoma_roc_auc']:.4f}" if pd.notna(row["melanoma_roc_auc"]) else "n/a"
        lines.append(f"| Run {index + 1} | {row['macro_f1']:.4f} | {row['melanoma_recall']:.2%} | "
                     f"{ap} | {auc} |")
    twenty = workloads[np.isclose(workloads["requested_review_fraction"], .2)]
    if len(twenty):
        lines += ["", "20% 검토 예산에서의 포착 건수(동일 건수 비교):", "",
                  "| Run | 전략 | 검토 건수 | 포착한 CNN 오분류 | 포착한 흑색종 오분류 | 남은 흑색종 오분류 |",
                  "|---|---|---:|---:|---:|---:|"]
        labels = {run: f"Run {i + 1}" for i, run in enumerate(classification["run_id"])}
        for _, row in twenty.iterrows():
            lines.append(f"| {labels[row['run_id']]} | {row['strategy']} | {int(row['n_reviewed'])} | "
                         f"{int(row['n_errors_reviewed'])} | {int(row['n_missed_melanoma_reviewed'])} | "
                         f"{int(row['n_missed_melanoma_unreviewed'])} |")
    lines += ["", f"퍼지 상태: **{metadata['fuzzy_status']}**.",
              "퍼지 점수가 없는 경우 CNN 기반 전략만 계산합니다. 임의의 퍼지 결과를 만들지 않습니다.", "",
              "[동일 검토 비율 비교 CSV](review_workload_curves.csv)의 각 행에 실제 검토 건수와",
              "포착한 오분류·흑색종 수, 검토하지 않고 남긴 흑색종 오분류 수가 있습니다.",
              "모든 전략은 같은 floor(N×비율) 건수를 선택하며, 동점은 이미지 ID로 정렬합니다.",
              "검토 대상으로 표시한 것은 정답으로 교정한 것이 아니므로 CNN 재현율을 바꾸지 않습니다.", "",
              "[흑색종 임계값 분석 CSV](melanoma_threshold_sweep.csv)는 별도의 이진 검토 플래그 분석입니다.",
              "7개 클래스 argmax 결과를 교체하거나 최종 임계값을 자동 선택하지 않습니다.", "",
              "확률은 이미 같은 검증셋에서 온도 보정됐을 수 있습니다. 이 보정 지표와 모델 선택 결과는",
              "독립 평가가 아닙니다. Bootstrap은 관측 클래스 내 병변 단위 재표집이며 환자 단위가 아닙니다.",
              "Bootstrap 구간은 분류 지표에만 적용하며 검토 포착률의 신뢰구간은 계산하지 않습니다.", "",
              "- [분류·흑색종 지표 CSV](classification_metrics.csv)",
              "- [클래스별 ROC-AUC·AP](class_probability_metrics.csv)",
              "- [검토 점수·순위 CSV](review_priority_scores.csv)",
              "- [전체 지표·bootstrap JSON](evaluation_details.json)",
              "- [흑색종 PR 곡선](01_melanoma_precision_recall.png)",
              "- [검토 비율별 비교 곡선](02_review_workload_comparison.png)",
              "- [Top-label 보정 곡선](03_top_label_reliability.png)",
              "- [평가 설정·퍼지 제공 여부](evaluation_metadata.json)", "",
              "그래프는 PNG와 SVG로 저장했습니다. 누락·중복된 퍼지 ID, 분할 불일치, 잘못된 확률은 거부합니다."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--fuzzy-scores", type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0, .05, .1, .2, .3, .5, 1])
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.bootstrap_resamples < 0 or args.bootstrap_resamples == 1:
        parser.error("Bootstrap resamples must be 0 (disabled) or at least 2")
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in args.budgets):
        parser.error("Budgets must be fractions in [0,1]")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    args.output_dir = (args.output_dir or ROOT / "reports/evaluation" / stamp).resolve()
    evaluate(args)


if __name__ == "__main__":
    main()

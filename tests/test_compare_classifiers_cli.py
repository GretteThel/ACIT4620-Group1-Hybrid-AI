"""Synthetic export contract test, not evidence of trained hybrid performance."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ham10000_hybrid.data_validation import CLASS_CODES
from ham10000_hybrid.metrics import compute_classification_metrics


def test_hybrid_cli_rejects_stale_checkpoint_then_compares_real_decisions(tmp_path):
    folder = tmp_path / "synthetic_cnn_fixture"
    folder.mkdir()
    y = np.tile(np.arange(7), 2)
    pred = y.copy()
    pred[1] = 0
    probs = np.eye(7)[pred] * .93 + .01
    ids = np.array([f"image_{i}" for i in range(14)])
    lesions = np.array([f"lesion_{i}" for i in range(14)])
    metrics = compute_classification_metrics([CLASS_CODES[i] for i in y], [CLASS_CODES[i] for i in pred])
    np.savez(folder / "validation_predictions.npz", image_id=ids, y_true=y, probs=probs)
    for name, value in {
        "run_status.json": {"state": "completed"},
        "experiment_config.json": {"class_codes": list(CLASS_CODES), "split_sha256": "fixture",
                                   "test_evaluated": False, "use_class_weights": False},
        "validation_metrics.json": metrics,
    }.items():
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    pd.DataFrame({"image_id": ids, "lesion_id": lesions, "label": y, "split": "validation"}).to_csv(
        folder / "split_manifest_summary.csv", index=False,
    )
    pd.DataFrame({"stage": ["fine_tune"], "epoch": [1], "train_loss": [.1], "val_loss": [.1],
                  **{f"val_{key}": [metrics[key]] for key in
                     ["accuracy", "balanced_accuracy", "macro_f1", "melanoma_recall"]}}).to_csv(
        folder / "training_history.csv", index=False,
    )
    model = b"synthetic checkpoint fixture; no model execution"
    (folder / "best_efficientnet_b0.pth").write_bytes(model)
    csv = tmp_path / "synthetic_hybrid.csv"
    pd.DataFrame({"image_id": ids, "predicted_dx": [CLASS_CODES[i] for i in y]}).iloc[::-1].to_csv(csv, index=False)
    metadata = {"source_cnn_run_id": folder.name, "source_checkpoint_sha256": "wrong",
                "split_sha256": "fixture", "class_codes": list(CLASS_CODES),
                "evaluation_split": "validation", "hybrid_fit_split": "train",
                "hybrid_version": "synthetic-contract-test-only"}
    sidecar = tmp_path / "synthetic_hybrid.json"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    out = tmp_path / "comparison"
    script = Path(__file__).resolve().parents[1] / "scripts/compare_classifiers.py"
    command = [sys.executable, "-X", "utf8", str(script), "--baseline-run", str(folder),
               "--hybrid-predictions", str(csv), "--hybrid-metadata", str(sidecar),
               "--bootstrap-resamples", "20", "--no-plots", "--output-dir", str(out)]
    rejected = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    assert rejected.returncode != 0
    assert "source_checkpoint_sha256" in rejected.stderr
    assert not out.exists()
    metadata["source_checkpoint_sha256"] = hashlib.sha256(model).hexdigest()
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "paired_comparison.json").read_text())
    assert report["metrics"]["accuracy"]["difference"] == pytest.approx(1 / 14)
    assert report["transitions"]["corrected_errors"] == 1
    assert report["transitions"]["introduced_errors"] == 0
    reports = json.loads((out / "classification_reports.json").read_text())
    assert reports["CNN + neuro-fuzzy"]["probability_metrics"] is None
    assert not (folder / "test_predictions.npz").exists()

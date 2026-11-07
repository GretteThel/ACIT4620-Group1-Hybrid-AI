"""Exercise the fuzzy score adapter with synthetic, isolated validation files."""

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


def test_cli_checks_fuzzy_provenance_and_exports_matched_budgets(tmp_path):
    pytest.importorskip("matplotlib")
    folder = tmp_path / "synthetic_fixture_run"
    folder.mkdir()
    y = np.tile(np.arange(7), 2)
    ids = np.array([f"fixture_{i}" for i in range(len(y))])
    probs = np.eye(7)[y] * .93 + .01
    lesions = np.array([f"lesion_{label}" for label in y], dtype=object)
    np.savez(folder / "validation_predictions.npz", y_true=y, probs=probs, image_id=ids,
             tta_probs=np.stack([probs, probs], axis=1), lesion_id=lesions)
    metrics = compute_classification_metrics([CLASS_CODES[i] for i in y], [CLASS_CODES[i] for i in y])
    for name, value in {
        "run_status.json": {"state": "completed"},
        "experiment_config.json": {"test_evaluated": False, "class_codes": list(CLASS_CODES),
                                   "split_sha256": "fixture_split", "use_class_weights": False},
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
    checkpoint_bytes = b"synthetic fixture; not a trained or loadable model"
    (folder / "best_efficientnet_b0.pth").write_bytes(checkpoint_bytes)
    fuzzy_path = tmp_path / "synthetic_fuzzy_scores.csv"
    fuzzy = pd.DataFrame({"run_id": folder.name, "image_id": ids,
                          "review_score": np.arange(len(y)), "checkpoint_sha256": "wrong",
                          "split_sha256": "fixture_split"})
    fuzzy.to_csv(fuzzy_path, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_review.py"
    out = tmp_path / "evaluation"
    command = [sys.executable, "-X", "utf8", str(script), str(folder), "--fuzzy-scores", str(fuzzy_path),
               "--bootstrap-resamples", "2", "--budgets", "0", ".5", "1", "--output-dir", str(out)]
    rejected = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    assert rejected.returncode != 0
    assert "different checkpoint_sha256" in rejected.stderr
    assert not out.exists()
    fuzzy["checkpoint_sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    fuzzy.iloc[::-1].to_csv(fuzzy_path, index=False)
    accepted = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    assert accepted.returncode == 0, accepted.stderr
    metadata = json.loads((out / "evaluation_metadata.json").read_text(encoding="utf-8"))
    assert metadata["fuzzy_evaluated_for"] == [folder.name]
    curve = pd.read_csv(out / "review_workload_curves.csv")
    half = curve[curve["requested_review_fraction"] == .5]
    assert half["n_reviewed"].eq(7).all()
    assert "fuzzy" in half["strategy"].tolist()
    assert (out / "01_melanoma_precision_recall.svg").is_file()
    assert not (folder / "test_predictions.npz").exists()

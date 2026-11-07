"""Validate saved-run comparisons independently of GPU libraries."""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = runpy.run_path(str(ROOT / "scripts/compare_runs.py"))
CODES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]


def save_run(folder, split="fixed_split"):
    folder.mkdir()
    truth = np.tile(np.arange(7), 2)
    ids = np.array([f"image_{i}" for i in range(len(truth))])
    probs = np.full((len(truth), 7), 0.01)
    probs[np.arange(len(truth)), truth] = 0.94
    np.savez(folder / "validation_predictions.npz", y_true=truth, image_id=ids, probs=probs)
    for name, value in {
        "run_status.json": {"state": "completed"},
        "experiment_config.json": {
            "test_evaluated": False, "class_codes": CODES, "split_sha256": split,
            "use_class_weights": False,
        },
        "validation_metrics.json": {key: 1.0 for key in FUNCTIONS["METRICS"]},
    }.items():
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    pd.DataFrame({"image_id": ids, "label": truth, "split": "validation"}).to_csv(
        folder / "split_manifest_summary.csv", index=False,
    )
    pd.DataFrame({
        "stage": ["fine_tune"], "epoch": [1], "train_loss": [0.1], "val_loss": [0.1],
        **{f"val_{key}": [1.0] for key in FUNCTIONS["METRICS"]},
    }).to_csv(folder / "training_history.csv", index=False)
    (folder / "best_efficientnet_b0.pth").write_bytes(b"fixture; never loaded as a model")
    return folder


def test_comparison_validates_predictions_and_preserves_epoch_data(tmp_path):
    a = FUNCTIONS["load_saved_run"](save_run(tmp_path / "a"))
    b = FUNCTIONS["load_saved_run"](save_run(tmp_path / "b"))
    FUNCTIONS["validate_comparison"]([a, b])
    assert a["computed"]["macro_f1"] == 1.0
    assert a["history"]["local_epoch"].tolist() == [1]
    assert a["history"]["checkpoint_chain_epoch"].tolist() == [1]


def test_comparison_rejects_different_split_and_ground_truth(tmp_path):
    a = FUNCTIONS["load_saved_run"](save_run(tmp_path / "a"))
    b = FUNCTIONS["load_saved_run"](save_run(tmp_path / "b", "different"))
    with pytest.raises(ValueError, match="different split_sha256"):
        FUNCTIONS["validate_comparison"]([a, b])
    b["config"]["split_sha256"] = "fixed_split"
    b["identity"].loc[0, "label"] = 5
    with pytest.raises(ValueError, match="different validation image IDs or labels"):
        FUNCTIONS["validate_comparison"]([a, b])


def test_comparison_rejects_invalid_probabilities(tmp_path):
    folder = save_run(tmp_path / "a")
    with np.load(folder / "validation_predictions.npz") as values:
        truth, ids, probs = values["y_true"], values["image_id"], values["probs"].copy()
    probs[0, 0] = np.nan
    np.savez(folder / "validation_predictions.npz", y_true=truth, image_id=ids, probs=probs)
    with pytest.raises(ValueError, match="Invalid validation probabilities"):
        FUNCTIONS["load_saved_run"](folder)


def test_comparison_rejects_metrics_not_matching_predictions(tmp_path):
    folder = save_run(tmp_path / "a")
    metrics = {key: 1.0 for key in FUNCTIONS["METRICS"]}
    metrics["macro_f1"] = 0.9
    (folder / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="Saved metric differs"):
        FUNCTIONS["load_saved_run"](folder)

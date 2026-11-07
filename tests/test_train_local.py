"""Check local-runner controls without importing PyTorch or accessing real data."""

import runpy
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "train_local.py"
FUNCTIONS = runpy.run_path(str(RUNNER))


def test_local_controls_keep_test_and_audit_protected(tmp_path):
    source = """from pathlib import Path
EXPERIMENT_NAME = 'old'
HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 2
REQUIRE_COMPLETE_AUDIT = False
RUN_FINAL_TEST = True
OUT = Path('/kaggle/working')
"""
    args = Namespace(output_dir=tmp_path, head_epochs=3, finetune_epochs=8, batch_size=32)
    namespace = {}
    exec(FUNCTIONS["configured_import_cell"](source, args), namespace)
    assert namespace["OUT"] == tmp_path
    assert namespace["REQUIRE_COMPLETE_AUDIT"] is True
    assert namespace["RUN_FINAL_TEST"] is False
    assert namespace["NUM_WORKERS"] == 0
    assert namespace["HEAD_EPOCHS"] == 3


def test_notebook_control_changes_stop_execution(tmp_path):
    args = Namespace(output_dir=tmp_path, head_epochs=3, finetune_epochs=8, batch_size=32)
    with pytest.raises(RuntimeError, match="Notebook controls changed"):
        FUNCTIONS["configured_import_cell"]("RENAMED_SETTING = 1", args)


def test_missing_inputs_record_failure_and_no_checkpoint(tmp_path):
    import json

    data = tmp_path / "empty_data"
    data.mkdir()
    out = tmp_path / "attempt"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(RUNNER),
         "--data-dir", str(data), "--output-dir", str(out)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert result.returncode == 1
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    inputs = json.loads((out / "input_check.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["mode"] == "dataset_training"
    assert len(inputs["missing_inputs"]) == 3
    assert (out / "run.log").is_file()
    assert not list(out.glob("*.pth"))


def test_existing_output_folder_is_not_overwritten(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("existing result", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--output-dir", str(tmp_path)],
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "existing result"
    assert not (tmp_path / "run_status.json").exists()


def test_warm_start_rejects_different_class_order_and_split():
    checkpoint = {"class_codes": ["nv", "mel"], "split_sha256": "split_a", "model_state": {}}
    validate = FUNCTIONS["validate_checkpoint_metadata"]
    validate(checkpoint, ["nv", "mel"], "split_a")
    with pytest.raises(ValueError, match="class order"):
        validate(checkpoint, ["mel", "nv"], "split_a")
    with pytest.raises(ValueError, match="different split"):
        validate(checkpoint, ["nv", "mel"], "split_b")


def test_warm_start_preserves_parent_selection_score():
    source = "history = []\nbest_score = (-1., -1.)\n"
    namespace = {}
    exec(FUNCTIONS["configured_training_cell"](source, (0.49, 0.46)), namespace)
    assert namespace["history"] == []
    assert namespace["best_score"] == (0.49, 0.46)
    with pytest.raises(RuntimeError, match="best-score initialization changed"):
        FUNCTIONS["configured_training_cell"]("renamed_best = None", (0.49, 0.46))


def test_head_training_cannot_be_skipped_without_checkpoint(tmp_path):
    out = tmp_path / "attempt"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--head-epochs", "0", "--output-dir", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "Skipping head training requires" in result.stderr
    assert not out.exists()


def test_warm_start_defaults_to_finetuning_only(tmp_path, monkeypatch):
    checkpoint = tmp_path / "saved_model.pth"
    checkpoint.write_bytes(b"parser fixture")
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--init-checkpoint", str(checkpoint)])
    args = FUNCTIONS["parse_args"]()
    assert args.head_epochs == 0
    assert args.finetune_epochs == 8
    assert args.init_checkpoint == checkpoint.resolve()

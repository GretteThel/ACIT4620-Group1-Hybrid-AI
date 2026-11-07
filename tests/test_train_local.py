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

"""Run the existing EfficientNet training notebook on local HAM10000 files.

Uses the notebook's training cells directly, retaining its fixed split and audit
gates. No images are downloaded, no splits are generated, and test evaluation
stays disabled. Every attempt gets a separate folder, including failed attempts.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/ACIT4620_02_EfficientNetB0_7Class_Training.ipynb"


class Tee:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, message):
        self.console.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--head-epochs", type=int, default=None)
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="Warm-start from a completed local run; optimizer and RNG are initialized anew",
    )
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4,
                        help="Stop a stage after this many epochs without validation improvement")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Training RNG seed; saved data splits stay fixed")
    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=False,
                        help="Use N_train/(7*class_count) cross-entropy weights from training only")
    parser.add_argument(
        "--check-environment", action="store_true",
        help="Only check pretrained model + GPU forward/backward with synthetic tensors",
    )
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("Training seed must be in [0, 2**32)")
    if args.head_epochs is None:
        args.head_epochs = 0 if args.init_checkpoint else 3
    if min(args.finetune_epochs, args.patience, args.batch_size, args.cpu_threads) < 1 or args.head_epochs < 0:
        parser.error("Fine-tuning epochs, patience, batch size and CPU threads must be positive")
    if args.head_epochs == 0 and not args.init_checkpoint:
        parser.error("Skipping head training requires --init-checkpoint")
    if args.init_checkpoint:
        args.init_checkpoint = args.init_checkpoint.resolve()
        if not args.init_checkpoint.is_file():
            parser.error("Initial checkpoint does not exist")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    kind = "environment_check" if args.check_environment else (
        "efficientnet_b0_weighted" if args.class_weights else "efficientnet_b0_baseline"
    )
    args.output_dir = (args.output_dir or ROOT / "reports" / "local_runs" / f"{stamp}_{kind}").resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def find_input(explicit, folder, filenames):
    if explicit:
        return explicit.resolve() if explicit.is_file() else None
    for name in filenames:
        candidates = sorted(folder.rglob(name))
        if len(candidates) > 1:
            raise ValueError(f"Multiple {name} files found; choose an explicit path")
        if candidates:
            return candidates[0].resolve()
    return None


def check_environment(out, cpu_threads, seed=42):
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a compatible CUDA PyTorch build")
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights).cuda()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7).cuda()
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.eval()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    inputs = torch.zeros(2, 3, 224, 224, device="cuda")
    targets = torch.tensor([0, 1], device="cuda")
    logits = model(inputs)
    nn.CrossEntropyLoss()(logits, targets).backward()
    if not torch.isfinite(model.classifier[1].weight.grad).all():
        raise RuntimeError("Non-finite gradients in GPU check")
    optimizer.step()
    torch.cuda.synchronize()
    report = {
        "purpose": "Environment check on synthetic tensors; NOT HAM10000 training or evaluation",
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "pretrained_weights": str(weights), "pretrained_url": weights.url,
        "output_shape": list(logits.shape), "forward_backward_optimizer_step": "passed",
        "dataset_used": False, "test_evaluated": False,
    }
    write_json(out / "environment_check.json", report)
    print(json.dumps(report, indent=2))


def configured_import_cell(source, args):
    """Override notebook controls structurally, leaving model/training code intact."""
    overrides = {
        "EXPERIMENT_NAME": args.output_dir.name,
        "HEAD_EPOCHS": args.head_epochs, "FINETUNE_EPOCHS": args.finetune_epochs,
        "PATIENCE": args.patience,
        "BATCH_SIZE": args.batch_size,
        "SEED": args.seed, "USE_CLASS_WEIGHTS": args.class_weights,
        # Notebook-defined dataset classes cannot be pickled by Windows spawn.
        "NUM_WORKERS": 0,
        "REQUIRE_COMPLETE_AUDIT": True, "RUN_FINAL_TEST": False,
    }
    tree = ast.parse(source)
    changed = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in overrides:
            node.value = ast.Constant(overrides[target.id])
            changed.add(target.id)
        elif target.id == "OUT":
            node.value = ast.parse(f"Path({str(args.output_dir)!r})", mode="eval").body
            changed.add("OUT")
    if changed != set(overrides) | {"OUT"}:
        raise RuntimeError("Notebook controls changed; update the local runner before executing")
    return compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec")


def configured_loss_cell(source):
    """Keep validation cross-entropy unweighted for fair plots across objectives."""
    tree = ast.parse(source)
    changed = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "evaluate_loader":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "criterion":
                    call.func = ast.Name(id="evaluation_criterion", ctx=ast.Load())
                    changed += 1
    if changed != 1:
        raise RuntimeError("Notebook validation loss changed; update the local runner")
    return compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec")


def validate_checkpoint_metadata(checkpoint, class_codes, split_sha256):
    if checkpoint.get("class_codes") != list(class_codes):
        raise ValueError("Initial checkpoint class order differs from this experiment")
    if checkpoint.get("split_sha256") != split_sha256:
        raise ValueError("Initial checkpoint uses a different split fingerprint")
    if "model_state" not in checkpoint:
        raise ValueError("Initial checkpoint is missing model_state")


def configured_training_cell(source, initial_score):
    tree = ast.parse(source)
    changed = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "best_score" for target in node.targets
        ):
            node.value = ast.parse(repr(tuple(initial_score)), mode="eval").body
            changed += 1
    if changed != 1:
        raise RuntimeError("Notebook best-score initialization changed; update the local runner")
    return compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec")


def load_initial_checkpoint(args, namespace):
    """Validate lineage and load weights; do not claim optimizer/RNG resumption."""
    torch = namespace["torch"]
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_metadata(checkpoint, namespace["CLASS_CODES"], namespace["SPLIT_SHA256"])
    parent = args.init_checkpoint.parent
    parent_status = json.loads((parent / "run_status.json").read_text(encoding="utf-8"))
    parent_settings = json.loads((parent / "experiment_config.json").read_text(encoding="utf-8"))
    if parent_status["state"] != "completed":
        raise ValueError("The source local run did not complete")
    if parent_settings["use_class_weights"] != namespace["USE_CLASS_WEIGHTS"]:
        raise ValueError("Warm-start comparisons must use the same loss weighting")
    for field, control in [
        ("seed", "SEED"), ("batch_size", "BATCH_SIZE"),
        ("backbone_lr", "BACKBONE_LR"), ("fine_tune_head_lr", "FINETUNE_HEAD_LR"),
    ]:
        if parent_settings[field] != namespace[control]:
            raise ValueError(f"Warm-start comparison changes {field}")
    digest = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
    parent_history = namespace["pd"].read_csv(parent / "training_history.csv")
    matches = parent_history.index[
        (parent_history["stage"] == checkpoint["stage"])
        & (parent_history["epoch"] == checkpoint["epoch"])
    ].tolist()
    parent_warm = parent_settings.get("warm_start")
    offset = int(parent_warm["cumulative_epochs_at_initialization"]) if parent_warm else 0
    if parent_warm and parent_warm.get("selected_checkpoint_origin") == "parent":
        source_epochs = offset
    elif len(matches) == 1:
        source_epochs = offset + int(matches[0]) + 1
    else:
        raise ValueError("Cannot identify the source checkpoint epoch in its history")
    namespace["model"].load_state_dict(checkpoint["model_state"], strict=True)
    lineage = {
        "source_checkpoint": str(args.init_checkpoint), "source_run": str(parent),
        "source_checkpoint_sha256": digest,
        "source_stage": checkpoint["stage"], "source_epoch": checkpoint["epoch"],
        "cumulative_epochs_at_initialization": source_epochs,
        "optimizer_state_restored": False, "rng_state_restored": False,
        "initialization": "saved model weights; fresh optimizer and seeded RNG",
    }
    namespace["local_warm_start"] = lineage
    write_json(args.output_dir / "warm_start.json", lineage)
    print("Loaded warm-start checkpoint:", args.init_checkpoint)
    print("Source checkpoint SHA-256:", digest)


@contextmanager
def keep_system_awake():
    kernel = None
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32
        if not kernel.SetThreadExecutionState(0x80000001):
            kernel = None
            print("Windows sleep-prevention request could not be enabled")
        else:
            print("Windows system sleep prevented for this run; display may sleep")
    try:
        yield
    finally:
        if kernel is not None:
            kernel.SetThreadExecutionState(0x80000000)


def train(args):
    manifest = find_input(args.manifest, args.data_dir, [
        "seven_class_master_manifest.csv", "ham10000_master_manifest.csv",
    ])
    audit = find_input(args.audit_summary, args.data_dir, ["audit_summary.json"])
    metadata = sorted(args.data_dir.rglob("HAM10000_metadata.csv"))
    missing = []
    if manifest is None:
        missing.append("Saved Notebook 01 master manifest with the existing split assignments")
    if audit is None:
        missing.append("Notebook 01 completed audit_summary.json")
    if len(metadata) != 1:
        missing.append("Exactly one HAM10000_metadata.csv and its original JPEG image folders")
    preflight = {
        "data_dir": str(args.data_dir),
        "manifest": str(manifest) if manifest else None,
        "audit_summary": str(audit) if audit else None,
        "metadata_files": [str(p) for p in metadata], "missing_inputs": missing,
    }
    write_json(args.output_dir / "input_check.json", preflight)
    if missing:
        raise FileNotFoundError("Training inputs are missing:\n- " + "\n- ".join(missing))

    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    from ham10000_hybrid.data_validation import CLASS_CODES, validate_manifest

    frame = pd.read_csv(manifest)
    validation = validate_manifest(frame)
    if "label" in frame and not frame["label"].eq(
        frame["dx"].map({code: i for i, code in enumerate(CLASS_CODES)})
    ).all():
        raise ValueError("Saved label indices differ from the project class order")
    write_json(args.output_dir / "manifest_validation.json", validation)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__local_training__"}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if code_cells[0].get("id") != "8232e368" or code_cells[1].get("id") != "fc72d83c":
        raise RuntimeError("Notebook structure changed; update the local runner")
    for index, cell in enumerate(code_cells):
        print(f"\nRunning notebook code cell {index + 1}/{len(code_cells)} ({cell['id']})")
        if index == 1:
            namespace.update({
                "DATA_DIR": metadata[0].parent, "MANIFEST_PATH": manifest,
                "manifest": frame,
                "pick_one": lambda pattern: [audit] if pattern == "audit_summary.json" else [],
            })
            continue
        source = "".join(cell["source"])
        code = configured_import_cell(source, args) if index == 0 else compile(
            source, f"{NOTEBOOK}:cell_{cell['id']}", "exec"
        )
        if cell["id"] == "6bfdf1b1":
            namespace["evaluation_criterion"] = namespace["nn"].CrossEntropyLoss()
            code = configured_loss_cell(source)
        if cell["id"] == "6709321c" and args.init_checkpoint:
            initial_metrics = namespace["evaluate_loader"](namespace["val_loader"])
            write_json(args.output_dir / "initial_validation_metrics.json", initial_metrics)
            shutil.copy2(args.init_checkpoint, args.output_dir / "best_efficientnet_b0.pth")
            code = configured_training_cell(
                source, (initial_metrics["macro_f1"], initial_metrics["balanced_accuracy"]),
            )
            print("Initial validation metrics:", initial_metrics)
        exec(code, namespace)
        if index == 0:
            namespace["torch"].set_num_threads(args.cpu_threads)
            if namespace["DEVICE"].type != "cuda":
                raise RuntimeError("No CUDA GPU available; full local training was not started")
            # Save notebook figures without opening GUI windows; release each figure.
            plt = namespace["plt"]
            plt.show = lambda *a, **kw: plt.close("all")
            namespace["display"] = print
            torch = namespace["torch"]
            write_json(args.output_dir / "runtime_environment.json", {
                "device": str(namespace["DEVICE"]), "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            })
        if cell["id"] == "2e69edb2" and args.init_checkpoint:
            load_initial_checkpoint(args, namespace)
        if cell["id"] == "6bfdf1b1":
            write_json(args.output_dir / "loss_configuration.json", {
                "training_objective": "class_weighted_cross_entropy" if args.class_weights else "cross_entropy",
                "class_codes": namespace["CLASS_CODES"],
                "training_class_counts": namespace["train_counts"].tolist(),
                "training_class_weights": namespace["class_weights"].tolist() if args.class_weights else None,
                "weight_source": "training split only",
                "validation_loss": "unweighted cross_entropy for all experiments",
                "training_loss_note": "Batch-reduced training objective; do not compare weighted/unweighted train-loss values directly",
            })
        if cell["id"] == "74ae9924":
            write_json(args.output_dir / "validation_metrics.json", namespace["val_metrics"])
            labels = namespace["val_y"]
            predictions = namespace["val_pred"]
            per_class = namespace["classification_report"](
                labels, predictions, labels=list(range(7)),
                target_names=namespace["CLASS_CODES"], output_dict=True, zero_division=0,
            )
            write_json(args.output_dir / "validation_classification_report.json", per_class)
    settings = namespace["settings"]
    settings["completed_epochs"] = len(namespace["history"])
    settings["patience"] = namespace["PATIENCE"]
    settings["validation_loss"] = "unweighted_cross_entropy"
    settings["training_class_weights"] = namespace["class_weights"].tolist() if args.class_weights else None
    settings["split_policy"] = "Reuse saved manifest; --seed changes training randomness only"
    if args.init_checkpoint:
        lineage = namespace["local_warm_start"]
        selected_digest = hashlib.sha256(
            (args.output_dir / "best_efficientnet_b0.pth").read_bytes(),
        ).hexdigest()
        lineage["selected_checkpoint_origin"] = (
            "parent" if selected_digest == lineage["source_checkpoint_sha256"] else "current_run"
        )
        settings["warm_start"] = lineage
        write_json(args.output_dir / "warm_start.json", lineage)
    write_json(args.output_dir / "experiment_config.json", settings)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "model_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".matplotlib"))
    os.environ["MPLBACKEND"] = "Agg"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False,
    ).stdout.strip()
    request = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items()}
    request.update({
        "git_commit": commit, "python": sys.version, "executable": sys.executable,
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_evaluated": False,
    })
    write_json(args.output_dir / "run_request.json", request)
    (args.output_dir / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True, capture_output=True, check=False,
    )
    if packages.returncode == 0:
        (args.output_dir / "environment_packages.txt").write_text(
            packages.stdout, encoding="utf-8",
        )
    status = {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "mode": "environment_check" if args.check_environment else "dataset_training"}
    write_json(args.output_dir / "run_status.json", status)
    failed = False
    with (args.output_dir / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print("Output folder:", args.output_dir)
            try:
                if args.check_environment:
                    check_environment(args.output_dir, args.cpu_threads, args.seed)
                else:
                    with keep_system_awake():
                        train(args)
                status["state"] = "completed"
            except Exception as error:
                failed = True
                status.update({"state": "failed", "error": str(error)})
                traceback.print_exc()
            finally:
                status["finished_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(args.output_dir / "run_status.json", status)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

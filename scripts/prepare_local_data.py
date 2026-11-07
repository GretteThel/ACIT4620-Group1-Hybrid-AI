"""Download public HAM10000 and run the original full audit notebook locally."""

import argparse
import ast
import hashlib
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

from train_local import ROOT, write_json

URL = "https://www.kaggle.com/api/v1/datasets/download/kmader/skin-cancer-mnist-ham10000"
NOTEBOOK = ROOT / "notebooks/01_HAM10000_Seven_Class_Data_Audit.ipynb"


def download(archive):
    if archive.is_file():
        print("Using existing archive:", archive, flush=True)
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=120) as response:
        if response.status != 206:
            offset = 0
        if "zip" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("Kaggle did not return a ZIP archive")
        total = offset + int(response.headers.get("Content-Length", "0"))
        received = offset
        last_update = time.monotonic()
        with partial.open("ab" if offset else "wb") as target:
            while chunk := response.read(8 * 1024 * 1024):
                target.write(chunk)
                received += len(chunk)
                if time.monotonic() - last_update >= 15:
                    print(f"Download: {received / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)
                    last_update = time.monotonic()
    if total and received != total:
        raise RuntimeError(f"Incomplete download: {received} bytes, expected {total}")
    partial.rename(archive)
    print("Download completed:", archive, flush=True)


def extract_original_images(archive, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        metadata = [info for info in source.infolist()
                    if Path(info.filename).name == "HAM10000_metadata.csv"]
        if len(metadata) != 1:
            raise RuntimeError("Expected exactly one source metadata CSV in the archive")
        selected = {}
        for info in source.infolist():
            path = Path(info.filename)
            if path.suffix.lower() != ".jpg":
                continue
            parts = [part for part in path.parts if part.lower() in {
                "ham10000_images_part_1", "ham10000_images_part_2",
            }]
            if not parts:
                continue
            canonical = parts[0].lower().replace("ham", "HAM", 1)
            target = Path(canonical) / path.name
            previous = selected.get(path.stem)
            if previous and (previous[0].CRC, previous[0].file_size) != (info.CRC, info.file_size):
                raise RuntimeError(f"Conflicting archive copies for image {path.stem}")
            selected.setdefault(path.stem, (info, target))
        if len(selected) != 10015:
            raise RuntimeError(f"Expected 10,015 distinct JPEGs, found {len(selected)}")
        files = [(metadata[0], Path("HAM10000_metadata.csv")), *selected.values()]
        for index, (info, relative) in enumerate(files):
            target = data_dir / relative
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing dataset file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Use canonical basenames, never archive-supplied traversal paths.
            with source.open(info) as incoming, target.open("wb") as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    outgoing.write(chunk)
            if index % 2000 == 0:
                print(f"Extracted {index}/{len(files)} files", flush=True)
    print("Extracted 10,015 original JPEGs and metadata (one copy per image ID)", flush=True)


class LocalAuditControls(ast.NodeTransformer):
    def __init__(self, data_dir, output):
        self.paths = {
            "/kaggle/input": str(data_dir),
            "/kaggle/working/ham10000_audit": str(output / "ham10000_audit"),
            "/kaggle/working/ham10000_7class": str(output / "ham10000_7class"),
        }

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in self.paths:
            return ast.copy_location(ast.Constant(self.paths[node.value]), node)
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(target, ast.Name) and target.id in {
            "RUN_FULL_IMAGE_AUDIT", "RUN_EXACT_HASH_AUDIT",
        } for target in node.targets):
            node.value = ast.Constant(True)
        return node


def audit(data_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "local_source.json", {
        "dataset_download_url": URL,
        "audit_notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "split_origin": "Reproduced from original Notebook 01, not imported Kaggle assignments",
        "image_copies": "One archive JPEG per image ID; duplicate folder copies omitted",
    })
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".matplotlib"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__local_audit__"}
    transformer = LocalAuditControls(data_dir, output)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        print(f"\nAudit notebook cell {index}: {cell['id']}", flush=True)
        tree = transformer.visit(ast.parse("".join(cell["source"])))
        exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec"), namespace)
        plt = namespace["plt"]
        plt.show = lambda *a, **kw: plt.close("all")
        namespace["display"] = print
        if cell["id"] == "539e62b1":
            expected = {
                "train": [4796, 782, 772, 358, 230, 89, 89],
                "validation": [974, 146, 168, 89, 34, 21, 9],
                "test": [935, 185, 159, 67, 63, 32, 17],
            }
            observed = namespace["split_by_class"]
            if any(observed[key].tolist() != counts for key, counts in expected.items()):
                raise RuntimeError(
                    "Split differs from recorded project counts. Install with "
                    "configs/local_training_constraints.txt; do not change the seed."
                )
    print("Completed full local audit:", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/downloads/ham10000.zip")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/ham10000")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data/notebook01")
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()
    if not args.skip_extraction:
        download(args.archive)
        extract_original_images(args.archive, args.data_dir)
    audit(args.data_dir.resolve(), args.audit_dir.resolve())


if __name__ == "__main__":
    main()

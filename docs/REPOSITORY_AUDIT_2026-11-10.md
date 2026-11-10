# Repository integration audit — 10 November 2026

## Scope and non-destructive status

Source: uploaded `ACIT4620-Group1-Hybrid-AI-master.zip`. This is a **snapshot audit**, not a full live Git-history review or a clinical validation. Its repository working tree and remote server were not modified. This cleanup package proposes **documentation-only** changes. Model checkpoints, experiments, images, generated outputs, original notebooks, freeze records, teammate methods and test splits are left untouched.

### Inventory (actual files in uploaded ZIP)

- **572 files**, approximately **178.29 MiB uncompressed**, **157.15 MiB ZIP**; directories are not counted as files.
- **17 `.ipynb` files**, **45 `.py` files**, **29 `.pth` files** (approximately **32.42 MiB**), **187 `.png` files**, **7 nested `.zip` files** (approximately **53.25 MiB**).
- **55 groups of byte-identical SHA-256 files**, with **72 duplicate occurrences** beyond one copy per group and approximately **22.39 MiB** of redundant bytes. These include *intentional cross-experiment reference copies*; counts are not a recommendation to delete all of them.
- The E2a-50 EfficientNet-B0 checkpoint is duplicated byte-for-byte between `experiments/e6_cnn_anfis_explainability/data/model_assets/` and `experiments/e7_computational_efficiency/data/model_assets/`, about **15.61 MiB each**. Multiple ANFIS checkpoint and validation CSV copies also exist.
- The largest nested bundle, `experiments/e6_cnn_anfis_explainability/outputs/ACIT4620_10_Explainability_Validation_Outputs.zip`, occupies ~**35.49 MiB**; extracted outputs also exist under that experiment. Its removal should be an **independent, reviewed decision** after verifying the ZIP's full coverage and the README's usage.

## Priority findings

### P0 — Notebook 12 fresh-clone reproducibility is incomplete

`evaluation/notebook12_locked_test/locked_test.py` imports `anfis_core` and verifies these four local paths:

```
evaluation/notebook12_locked_test/anfis_core.py
evaluation/notebook12_locked_test/data/e3_all_seed_metrics.csv
evaluation/notebook12_locked_test/data/e4_all_seed_metrics.csv
evaluation/notebook12_locked_test/data/validation_selected_checkpoints.csv
evaluation/notebook12_locked_test/data/sealed_split_manifest.zip
```

The uploaded snapshot contains **none of these** in the Notebook 12 directory. It contains other ANFIS implementations and validation CSVs in experiment folders, **but they must not be substituted blindly**. The pinned SHA-256 values in `locked_test.py` for E4 metrics and the validation-selection CSV **do not match** the superficially corresponding E4/E5 CSVs in this uploaded snapshot. E3 metrics do match. An explicitly verified, frozen-source repair is needed; do **not** change expected hashes or re-freeze to hide a mismatch. Keep the original successful local freeze and its evidence. The sealed manifest contains test partition labels; **do not publish it** merely to make the GitHub tree self-contained. Team sign-off is still required before the final test is opened.

### P1 — Root README and some methodological documents are stale

The former root README was dated **7 November 2026** and stated that neuro-fuzzy classification had not been implemented, despite the completed E4 ANFIS, E5 comparison, E6 explanation, E7 efficiency and Streamlit app included in the snapshot. It also described teammate local CNN runs as if they were the entire current project state. It has been preserved *verbatim* at `docs/README_HISTORY_2026-11-07.md`; this kit replaces only the root overview, not the original teammate reports. Documents `docs/methodology.md`, `docs/experiment_protocol.md`, `docs/neuro_fuzzy_evaluation.md`, `docs/team_workflow.md`, and `src/README.md` retain older prospective framing. Treat these as **historical design documents** pending collaborative review, rather than present-tense statements about all completed work. The team workflow even advises using `main`, while the actual repository uses `master`.

### P1 — Data and model asset policy is internally inconsistent

Root `.gitignore` excludes newly added `.pth` files, but the ZIP already contains **29 tracked `.pth` files**, including the two identical EfficientNet backbones. The app README says the app distribution is source-only, which is true of `artifact/cnn_anfis_streamlit`, **not of the repository as a whole**. Also, some committed Grad-CAM overlay PNGs contain visual information from original dermoscopic images, even though raw HAM10000 JPEG files were not found in the snapshot. Confirm project/dataset rights before changing repository visibility.

### P2 — Avoid duplicated bundles and unstable path references

Some `outputs/*.zip` files bundle extracted results already committed adjacent to them. `experiments/e7_computational_efficiency/outputs/` contains both the original and corrected-memory result ZIP. Preserve the **corrected** benchmark interpretation; do not cite sampled process RSS as an exact model-only peak. The E3, E5 and Streamlit READMEs include local Windows examples or external download links; these should eventually become parameterized setup instructions, without changing scientific data provenance.

### P2 — Notebook naming is deliberate

E3 / E4 / E5 / E6 / E7 are **experiment numbers**, corresponding to Notebooks **07 / 08 / 09 / 10 / 11**. Notebooks 01–06 and 03B–03D are in `notebooks/`; E3–E7 are in `experiments/`; Notebook 12 is under `evaluation/`. Do not rename/move the working experiment folders merely to force consecutive numbering.

## Testing performed for this snapshot

- `PYTHONPATH=src python -m pytest -q tests`: **48 passed** in the audit environment.
- `python -m unittest discover -s artifact/cnn_anfis_streamlit/tests -v`: **16 passed**.
- The separate Notebook 12 unit-test package was **not** independently runnable from this ZIP without restoring its pinned assets and `anfis_core` import; this is documented as a limitation, not papered over by changing the code.
- This is **not** an exhaustive security, secret-scanning or notebook execution audit; no original HAM10000 images, held-out test set or training was opened.

## Safe immediate actions (proposed PR scope)

1. Replace only the main `README.md` with the factual integration overview in this kit.
2. Preserve the 7 November original at `docs/README_HISTORY_2026-11-07.md`.
3. Add this audit under `docs/REPOSITORY_AUDIT_2026-11-10.md`.
4. Make a separate follow-up issue / PR to repair Notebook 12's **hash-verified** dependencies using the original local experiment inputs, without invalidating the pre-test freeze or revealing the sealed manifest.
5. Discuss whether to consolidate duplicated source assets/ZIP exports *after checking downstream imports, existing experiment READMEs, and licensing*. This is **not** part of the current safe PR.
6. After genuine team sign-off and once-run locked-test evaluation, update the current README's status and integrate evidence into the IEEE report. Do not invent an ANFIS performance improvement or claim acquisition-artifact subgroup conclusions from machine-only prescreening.

## No automatic deletions

The audit intentionally makes **no file deletion recommendations executable without human review**. In particular, do not git-rm all `.pth`, `.npz`, `.png`, `.csv` or `outputs/*.zip` in bulk: many are provenance-linked to fixed experiment results and validations.

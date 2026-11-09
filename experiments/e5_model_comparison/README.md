# ACIT4620 Group 1 — Notebook 09: Matched Validation Model Comparison

This folder is ready to open in **Windows VS Code**. It contains the full notebook, reusable script, **actual prediction CSVs** from completed CNN E2a-50, E3 Linear/MLP and E4 ANFIS experiments, a provenance manifest, and already generated validation analysis outputs.

**No Kaggle / Colab account, GPU, HAM10000 raw images, CNN checkpoint, or retraining required.** The test set is not accessed.

## 1. Open in VS Code

Open the **`ACIT4620_09_Model_Comparison_VSCode`** folder (this folder, not its parent). Open `ACIT4620_09_Model_Comparison_VSCode.ipynb` and select your **existing E3/E4 `.venv` Python kernel**. You successfully used PyTorch, NumPy, pandas, scikit-learn and matplotlib in that environment.

Run notebook cells from top to bottom. For a terminal-only alternative run:

```powershell
# From the Notebook 09 folder, using the existing E3 virtual environment:
& "C:\Users\grela\OneDrive\Desktop\OSLOMET\COMPUTATIONAL INTELLIGENCE\FOR FINAL REPORT\CODE\ACIT4620_E3_VSCode_Complete\.venv\Scripts\python.exe" .\model_comparison.py
```

If that Python environment is in a different location on your computer, replace the interpreter path accordingly. You can also execute `python model_comparison.py` when your E3 `.venv` is activated.

**Input files are included in `inputs/`**, so you do not need to manually copy the 12 MB ANFIS ZIP or the 1.5 MB E3 ZIP into this folder.

## 2. Scientific protocol

- Same seven-class HAM10000 lesion-disjoint split and exact split fingerprint `265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0`.
- Exactly **1,441 validation images** from **1,067 lesions**; image, lesion and true-label IDs are checked across 16 prediction exports (CNN 1 + E3 6 + E4 9).
- E3 and E4 shared the **E2a-50 frozen EfficientNet-B0 features**, verified by source checkpoint hash.
- Recalculate all metrics from original-view probability columns. Small numerical rounding differences are normalized **only in memory** for NLL/AUC/AP; input CSVs are never overwritten.
- Select best seed separately within each architecture/rule-count family by validation macro-F1, while also publishing three-seed means and sample SDs.
- Exploratory 1,200-resample **class-stratified lesion-level paired bootstrap** for selected comparisons. Confidence intervals are conditional on these fitted and selected models, not independent test evidence, and do not include training-seed or selection uncertainty.
- Do not access the held-out **1,458-image test set**. No claim of a new test accuracy.

## 3. Primary results reproduced on real outputs

| Selected model | Validation accuracy | Macro-F1 | Balanced accuracy | Melanoma recall |
|---|---:|---:|---:|---:|
| E2a-50 CNN | 0.7321 | 0.5500 | 0.6217 | 0.4315 |
| Linear, seed 42 | 0.7065 | 0.5455 | 0.6317 | 0.4452 |
| MLP, seed 42 | 0.7550 | **0.5749** | 0.5832 | 0.4247 |
| ANFIS 4, seed 52 | 0.7446 | 0.5499 | 0.5748 | 0.4315 |
| ANFIS 8, seed 42 | 0.7349 | 0.5588 | 0.5989 | 0.3699 |
| ANFIS 16, seed 52 | 0.7266 | 0.5676 | 0.6205 | 0.4521 |

Mean macro-F1 across seeds: Linear **0.5417**, MLP **0.5605**, ANFIS-4 **0.5462**, ANFIS-8 **0.5509**, ANFIS-16 **0.5430**. The 16-rule ANFIS had the best individual ANFIS seed but highest across-seed variation.

Our local reproducible 1,200-resample bootstrap (ANFIS-16 seed 52 **minus** MLP seed 42) is **−0.0073**, approximate 95% CI **[−0.0533, +0.0485]**. The interval includes zero. It does **not prove equivalence**, and does not support claims that ANFIS generally outperforms the MLP.

## 4. What it answers

- **RQ1:** Our data-centric experiments (E1/E2a/E2b) are important context, but this notebook does not isolate their causal effects.
- **RQ2:** Matched validation comparison of the CNN, linear, MLP and ANFIS heads: competitive ANFIS results, not uniform superiority; identify accuracy/sensitivity trade-offs.
- **RQ3:** E4 rule activation/contribution/ablation exists but image-level faithfulness, occlusion and Grad-CAM require a subsequent experiment.
- **RQ4:** Reports parameter counts and recorded E3/E4 head-only training costs, not end-to-end latency, throughput or power consumption; separate matched benchmarks needed.

## 5. Output artifacts

`outputs/ACIT4620_09_Validation_Comparison/` includes CSVs for every model, selected checkpoints, class metrics, paired error overlap, lesion bootstrap CIs, recorded head training costs, three PNG figures, and `evaluation_metadata.json` with source hashes. An additional `outputs/ACIT4620_09_Validation_Comparison_Outputs.zip` is generated when you run the script.

### GitHub

After verifying notebook outputs, copy this entire folder under `experiments/e5_model_comparison/` in your team's GitHub repository, on a separate feature branch. **Do not add the original images, unrelated private files, or any train/test split regeneration.** This package uses provenance-preserving input copies; the E3 original configuration had a local user path replaced with a redaction placeholder.

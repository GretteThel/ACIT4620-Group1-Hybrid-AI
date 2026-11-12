# ACIT4620 Group 1 — Notebook 08: E4 First-Order Takagi–Sugeno ANFIS

## What this package implements

This is the **direct multiclass first-order Takagi–Sugeno ANFIS head** specified in the submitted Group 1 proposal, **not** Mamdani post-processing or a learned correction of CNN probabilities.

Frozen E2a-50 pooled EfficientNet-B0 **1,280-D** features → learned **linear projection 1,280 → 8** → **R=4, 8, 16** fuzzy Gaussian-rule premises → stable normalized firing strengths → **seven independent first-order affine consequents per rule** → weighted aggregation into **seven class logits** → Softmax.

The 8-dimensional projection, positive-width Gaussian premises, and consequent coefficients are jointly trained using **Adam and backpropagation**. The original proposal explicitly distinguishes this from classical ANFIS least-squares + gradient-descent hybrid training. KMeans on the **training partition only** initializes the centers, but does not change the network architecture or fit on validation/test.

## 1. Set up locally in Windows VS Code

1. Extract `ACIT4620_08_E4_ANFIS_VSCode.zip`.
2. In VS Code select **File → Open Folder** and select the folder containing **`train_e4_anfis.py` and `requirements.txt`**, rather than its outer parent.
3. Copy the **existing, extracted** `ACIT4620_shared_features_e2a50` folder from our E3 project's `data/` into this E4 project's `data/`. Alternatively, put the original `ACIT4620_06_Shared_Features.zip` inside `data/` and the script will unzip it locally. Do **not** copy the E3 results ZIP as a substitute for Notebook 06 features.
4. Open **Terminal → New Terminal** and verify `Get-ChildItem requirements.txt,train_e4_anfis.py` shows both files.

Using the installed Python 3.13, run one command per line:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tests\test_anfis_core.py
```

You can **reuse the already working E3 virtual environment** by selecting its Python executable as the VS Code notebook kernel, rather than reinstalling packages; the E4 package's requirements are covered by the E3 environment.

## 2. Run the notebook OR the script, not both

**Notebook:** open `ACIT4620_08_E4_ANFIS_VSCode.ipynb`, choose a local Python kernel, and run cells from top to bottom. It has the full model code embedded and does not depend on Kaggle/Colab or the external `anfis_core.py` file.

**Script:** in the E4 project folder, run:

```powershell
.\.venv\Scripts\python.exe train_e4_anfis.py
```

If using the E3 environment instead of creating a local `.venv`, substitute the exact path to the E3 environment's `python.exe`.

**No GPU is required** because CNN features were already extracted. If a compatible CUDA GPU is available, PyTorch may use it; CPU execution is fully supported. Do not turn off the computer mid-experiment unless outputs are backed up.

## 3. Fixed protocol (identical inputs to E3)

- HAM10000: **seven diagnoses** `[nv, mel, bkl, bcc, akiec, vasc, df]`.
- Frozen E2a-50 checkpoint SHA-256: `2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9`.
- Split fingerprint SHA-256: `265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0`.
- **Training:** 7,116 images, **validation:** 1,441 images; validation is 1,067 lesions.
- All **1,458 test images** are locked, not read or evaluated here.
- Training-only feature-standardization and inverse-frequency class-weighted multiclass cross-entropy; no validation leakage.
- R = **4, 8, 16** rules × seeds **42, 52, 62** = **nine training runs**.
- Maximum **100 epochs**, batch size **128**, learning rate **0.001**, weight decay **0.0001**, gradient clip **5**, early-stopping patience **15**, validation macro-F1 selects the saved checkpoint.
- **Optimizer:** Adam as specified in the submitted ANFIS proposal; E3 used AdamW, so any comparison should describe this implementation difference instead of claiming all optimizer choices were isolated.
- **Validation metrics:** macro-F1 (primary), accuracy, balanced accuracy, melanoma precision/recall, per-class metrics, macro OvR AUROC, macro OvR average precision, Brier score and NLL. Results are **uncalibrated original-view probabilities**.

Trainable parameter counts, including the 1,280 → 8 projection:

| Model | Trainable parameters |
|---|---:|
| E3 MLP head | 10,767 |
| E4 ANFIS R=4 | 10,564 |
| E4 ANFIS R=8 | 10,880 |
| E4 ANFIS R=16 | 11,512 |

## 4. Outputs

`outputs/ACIT4620_E4_ANFIS/` will contain:

- `e4_all_seed_metrics.csv`, `e4_seed_summary.csv`, `e4_validation_selected_by_rule_count.csv` and selection/config JSON files.
- Nine per-run folders such as `anfis_r8_seed42/` containing saved ANFIS checkpoints, validation probabilities and class labels, confusion matrices, training histories, class-wise results, normalized rule strengths, signed class-wise contributions, Gaussian centers/widths and controlled rule-ablation outputs.
- The final backup `outputs/ACIT4620_E4_ANFIS_Outputs.zip` (size depends on results).

Rule ablation holds learned parameters fixed, removes the most/least absolute signed-contribution rule for each predicted class, and **renormalizes** the remaining firing strengths. This is a narrow, documented **latent-rule faithfulness proxy**. It is not image-level occlusion, image-level Grad-CAM, or evidence of medically meaningful fuzzy antecedents.

End-to-end CNN+ANFIS inference latency and ANFIS-aware Grad-CAM are **subsequent** image-level experiments: they require loading the original E2a-50 CNN checkpoint and HAM10000 images, passing gradients from an ANFIS class logit back to CNN feature maps. This notebook measures **head-only** inference cost and tracks rule-analysis/export time separately.

## 5. Troubleshooting

- `No such file ... requirements.txt`: terminal is in the **wrong directory**. Run `Get-Location` and `Get-ChildItem` and enter the folder containing `requirements.txt`.
- `Notebook 06 features missing`: move `ACIT4620_shared_features_e2a50/` or `ACIT4620_06_Shared_Features.zip` into the E4 `data/` folder. The program deliberately refuses to create a new random split.
- `Expected one Notebook 06 feature source; found 2`: remove the duplicate extracted feature folder, retaining the verified single source.
- `Checkpoint hash mismatch`: the selected E2a-50 feature source is not the same one used by E3. **Do not bypass** the check.
- GPU unavailable / `CUDA False`: CPU is supported.
- If results show unexpectedly low macro-F1, inspect class-specific outputs, rule activations and the recorded training history; **do not claim** neuro-fuzzy superiority without evidence.

## Optional validation-only weight-decay comparison

The original defaults and saved outputs remain the reference experiment.
For a separate sensitivity experiment, use the original Notebook 06 input ZIP
or its extracted feature folder:

```powershell
..\..\.venv\Scripts\python.exe run_weight_decay_sweep.py --features 'C:\path\to\ACIT4620_06_Shared_Features.zip'
```

Run this command from `experiments/e4_sugeno_anfis`. The fixed comparison is
`weight_decay = [1e-4, 3e-4, 1e-3]`, sixteen rules, and seeds `[42, 52, 62]`:
nine training runs. Learning rate, batch size, maximum epochs, early stopping,
feature standardization and model architecture retain the reference settings.

Use `--dry-run` to inspect the plan without reading data or writing outputs.
An optional `--output-dir` must name a directory that does not already exist.
By default, new results go under `reports/e4_weight_decay_<UTC timestamp>/`.
The comparison includes all per-run outputs, matched-seed differences from the
fresh baseline, a mean/sample-SD summary CSV, and `weight_decay_comparison.png`.
Original E4 results and the locked-test model selection are not replaced.

The training script also accepts `--weight-decay`, `--rule-counts`, `--seeds`,
`--features` and `--output-dir` for individual exploratory runs. With no options,
its original experiment settings and output location are retained. Use a new
explicit output directory for all additional experiments.

## 6. GitHub hygiene

For the team repository audit, **all 150 E4 output files and the original 12.33 MB E4 output ZIP are intentionally included** in this isolated E4 folder, per the team's decision. This is feasible for the present artifact sizes but increases Git storage. The repository root ignores `*.npz` and `*.pth`; use the project-specific force-add command in `PUSH_E4_TO_GITHUB.md` to include the nine model checkpoints and all explanation arrays. Do not commit raw HAM10000 images, input feature datasets, environments, or credentials.

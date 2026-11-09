# ACIT4620 E3 — complete local VS Code edition

This edition avoids Kaggle and Colab sessions. It trains only lightweight classification heads from the frozen 1,280-dimensional CNN features already extracted in Notebook 06. **It does not train EfficientNet again.** It never opens the locked test set.

## 1. Obtain the input feature ZIP

On your computer, open your existing Drive file URL:

https://drive.google.com/file/d/1t4ZEkqlaFs1QbH-tfZMQDeyFG3gmzxW1/view?usp=drive_link

Download the large (~38.7 MB) **`ACIT4620_06_Shared_Features.zip`**. Put it into this project's `data/` folder. It can also remain in your computer's Downloads folder under its original filename. This is distinct from the small ZIP containing the current source code.

## 2. Install VS Code tools

Install Python 3.11 or 3.12 and VS Code. In the VS Code Extensions panel install **Python** and **Jupyter** (Microsoft). Open this extracted project folder in VS Code using File → Open Folder.

## 3. Create and select a virtual environment on Windows

In VS Code's Terminal (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.11` is unavailable but Python is installed, use `python -m venv .venv` instead. Installation requires internet once. The experiment itself does not need an internet session.

Open `ACIT4620_07_E3_Linear_MLP_VSCode.ipynb`, click **Select Kernel**, select `.venv` and **Run All**.

**Alternative without Jupyter:**

```powershell
.\.venv\Scripts\python.exe train_e3_vscode.py
```

Note: if a `ModuleNotFoundError` occurs despite installation, ensure the selected notebook kernel is exactly the `.venv` Python where the packages were installed.

## 4. Input verification

The notebook checks the original class order `nv, mel, bkl, bcc, akiec, vasc, df`, split fingerprint `265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0`, shapes train `(7116, 1280)` and validation `(1441, 1280)`, lesion-disjoint identity, and train-only feature normalization. All must pass.

## 5. Outputs

The script trains a linear head (1280→7) and MLP (1280→8→32→7), seeds 42, 52, 62, class-weighted cross-entropy, AdamW, max 100 epochs, patience 15, and selection by validation macro-F1. CPU is supported, and CUDA is used only if PyTorch detects it. On successful completion, it writes:

- `outputs/ACIT4620_E3_Linear_MLP/e3_all_seed_metrics.csv`
- `outputs/ACIT4620_E3_Linear_MLP/e3_seed_summary.csv`
- `outputs/ACIT4620_E3_Linear_MLP/e3_validation_selected_heads.csv`
- `outputs/ACIT4620_E3_Linear_MLP/train_feature_scaler.npz`
- Six per-head/seed folders with checkpoints, confusion matrices, training history, predictions, and metrics
- `outputs/ACIT4620_E3_Linear_MLP_Outputs.zip`

Keep the ZIP for sharing or later comparison with the E4 ANFIS experiment. The raw images and original E2a-50 checkpoint are **not** needed for this stage.

## 6. Troubleshooting

- **Cannot find feature ZIP:** Place `ACIT4620_06_Shared_Features.zip` in `data/`, or set `ZIP_OVERRIDE` in the first notebook cell or in the script.
- **Wrong ZIP:** If it reports a missing `feature_extraction_config.json`, you likely downloaded the source-code ZIP rather than the Notebook 06 feature archive.
- **CPU only:** That is fine. It may be slower than GPU but requires no cloud runtime.
- **Many folders:** If both an extracted feature folder and a second duplicate are present inside `data/`, keep only one to avoid ambiguous discovery.
- **Don't commit data:** `.gitignore` excludes `data/`, `outputs/` and model binaries.

# ACIT4620 Group 1 · CNN–ANFIS research demonstration

This repository-ready Streamlit application runs the **original frozen** EfficientNet-B0 (1,280 pooled features) with training-only feature standardization, a trainable 1,280→8 projection, and a 16-rule first-order Sugeno ANFIS (seed 52) for the seven HAM10000 classes. The trained checkpoints remain unchanged.

**Research use only.** Not a medical device, diagnosis, triage or melanoma screening service. Softmax scores are uncalibrated model outputs, not cancer risk.

## Verify and launch on Windows (existing project)

1. Copy source files into the existing `ARTIFACT\ACIT4620_CNN_ANFIS_App` application directory; the ZIP intentionally contains **no model weights**.
2. Keep the three original files in the app's `model_assets` directory, or restore using `setup_model_assets.ps1` with the original Notebook 11 `data\model_assets` folder.
3. Use the same working Python virtual environment as the earlier E3/Notebook12 experiments.

```powershell
$base = 'C:\Users\grela\OneDrive\Desktop\OSLOMET\COMPUTATIONAL INTELLIGENCE\FOR FINAL REPORT'
$app = "$base\ARTIFACT\ACIT4620_CNN_ANFIS_App"
$py = "$base\CODE\ACIT4620_E3_VSCode_Complete\.venv\Scripts\python.exe"
Set-Location $app
& $py -m pip install -r .\requirements-app.txt
& $py -m unittest discover -s .\tests -v
& $py .\run_inference.py check --assets .\model_assets
& $py -m streamlit run .\app.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://127.0.0.1:8502` and upload **authorized validation or demo JPEG/PNG images only**. Do not use the untouched held-out test set before genuine group sign-off and the prespecified one-time evaluation. Stop the Streamlit process with `Ctrl+C`.

## Verify seven different actual validation classes (read-only)

From the same app directory, in a separate terminal:

```powershell
$nb10 = "$base\CODE\Notebook_10_CNN_ANFIS_XAI_VSCode_Complete\ACIT4620_10_CNN_ANFIS_XAI_VSCode"
& $py .\validation_parity_seven_classes.py `
    --assets .\model_assets `
    --reference "$nb10\data\reference\validation_anfis_r16_seed52.csv" `
    --raw "$nb10\data\raw"
```

This deterministically selects one validation image from **each true diagnostic class** (seven images, not seven correct predictions), verifies its model prediction and seven softmax scores against the saved original validation export, and refuses non-validation reference filenames or unexpected sample sizes. Some selected original examples may be misclassified; that does not constitute a code failure. This is an engineering reproducibility check, **not an additional model accuracy study or test evaluation**.

## What's changed in this version

- Replaced the small-number, all-zero-labelled vertical rule-contribution chart with a **horizontal, zero-anchored Altair chart**. It preserves sign and magnitude of original logit contributions and uses labelled numeric ticks (not percentages).
- Preserved the earlier corrected, side-by-side **0–100%** model-score comparison and readable static tables.
- Added tested `src/chart_utils.py` and `validation_parity_seven_classes.py` for a deterministic seven-class validation-only parity check.
- No changes to CNN/ANFIS weights, model preprocessing, scalers, Grad-CAM logic, validation scores, or Notebook 12's pre-test freeze.

## GitHub

See `PUSH_TO_GITHUB.md`. The distribution intentionally omits checkpoints, raw images, processed predictions, and local paths to test data. The repo stores **code only**; each authorized team member restores locally held original checkpoints with verified hashes using `setup_model_assets.ps1`.

## Limitations

- Model explanations are not causal ground truth or lesion segmentations.
- Rule contributions operate on logits, not class probabilities. Dominance by Rule 7 in one example does not establish dataset-wide collapse.
- No held-out test results have been reported. Group sign-off is still required before executing the sealed evaluation.
- The separate acquisition-artifact human-review study is pending and is not claimed as complete.

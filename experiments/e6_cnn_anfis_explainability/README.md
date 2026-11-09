# ACIT4620 Group 1 — Notebook 10: ANFIS-aware Grad-CAM and image occlusion

**Purpose:** Complete the image-level part of research question **RQ3** in the submitted proposal: *How stable and faithful are class-specific Grad-CAM explanations and signed ANFIS rule contributions, especially for melanoma false negatives and images with acquisition artifacts?*

## Precisely what is implemented

**Original image → ImageNet-normalized 224×224 input → frozen, validation-selected E2a-50 EfficientNet-B0 → pooled 1,280 features → training-only feature scaling → eight-dimensional projection → 16-rule, seven-output first-order Takagi–Sugeno ANFIS**. The alternative original CNN classifier head is run on the **same spatial feature maps**.

The notebook computes **Grad-CAM from the pre-Softmax class logit** through the *entire frozen CNN–ANFIS inference path*. The ANFIS output is not a post-hoc adjustment of the CNN classifier probabilities. For melanoma false negatives, it explains the **true melanoma logit** and separately the **incorrect predicted-class logit**. It also provides the **signed ANFIS rule contributions** for each chosen class.

Additional evaluation: heatmap stability between 16-rule ANFIS seeds 42/52/62; optional horizontal-flip consistency; and equal-area, blurred-patch deletion of top-ranked, bottom-ranked and random regions. **These are explanatory diagnostics, not a clinical or causal proof.**

## Included in this package

- `ACIT4620_10_Explainability_VSCode.ipynb` — Notebook 10 (run cells top to bottom)
- `hybrid_xai.py` — complete reusable integrated inference, explanation, and evaluation implementation
- `anfis_core.py` — **unchanged** E4 first-order Sugeno implementation
- `tests/test_hybrid_xai.py` — reproducible real-checkpoint smoke test on a random tensor (not a clinical result)
- `data/model_assets/` — real E2a-50 CNN checkpoint, trained 16-rule ANFIS heads for seeds 42/52/62, and train-only E4 scaler
- `data/reference/` — the actual E2a-50 and selected E4 validation prediction exports for checking exact reconstruction
- `data/artifact_annotations.csv` — **empty template** (not invented artifact annotations)
- `data/raw/` — **place your locally obtained original HAM10000 image folders here**

The CNN file is SHA-256 verified against `2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9`. CNN and ANFIS checkpoints share the same lesion-disjoint split SHA-256 `265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0`. The selected E4 head is **16 rules / seed 52**; seeds 42 and 62 are used for stability, not re-selection.

## Setup in VS Code on Windows

1. Extract the complete ZIP and open the inner `ACIT4620_10_CNN_ANFIS_XAI_VSCode` folder using **VS Code → File → Open Folder**.
2. Download **Skin Cancer MNIST: HAM10000** from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 . Extract **both** `HAM10000_images_part_1` and `HAM10000_images_part_2` into `data/raw/`. All 10,015 original image filenames may exist locally, **but the code loads only the image IDs in the pre-saved validation selection**; it never loads test images or labels.
3. In the project folder, choose your existing compatible Python environment. If the packages are missing, run `python -m pip install -r requirements.txt` in that **selected** environment (the notebook requires `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `Pillow`, and `matplotlib`). CPU works.
4. Open `ACIT4620_10_Explainability_VSCode.ipynb`, **Select Kernel**, and run the cells **in order**. The checkpoint loader and the first validation-image prediction reconstruction must pass before Grad-CAM begins. If the reconstruction fails, **stop** rather than weakening the tolerance or interpreting mismatched gradients.

To do a code-only smoke test **without raw images** (this is NOT a real experiment):

```powershell
python hybrid_xai.py --smoke-test
```

To run the entire experiment from the project folder after placing original images:

```powershell
python hybrid_xai.py --max-images 13 --device cpu
```

## Optional independent artifact annotations

The proposal asks about acquisition artifacts, but the available E4 prediction data **do not contain reliable human artifact labels**. You may fill `data/artifact_annotations.csv` with verified validation image IDs and a documented observation source:

```csv
image_id,artifact_type,annotation_source
ISIC_00xxxxxx,hair,manual_review_v1
```

Do not invent annotations. Without independent labels, report only overall or melanoma-error explanation observations; say that the artifact-stratified RQ3 analysis is pending.

## Default analysis budget

- 13 deterministic lesion-distinct validation images (melanoma FN/TP and six other classes)
- Original CNN and selected ANFIS Grad-CAM using **the same class target and image**
- For melanoma false negatives, an additional predicted-class Grad-CAM
- Signed per-rule contributions for the target logit, with class-logit reconstruction check
- Seed-variation and optional horizontal-flip saliency comparison on the first six images
- Tiled image occlusion on eight validation images, 10%, 20%, 30% tiles masked, compared with bottom/random controls

This is a **purposively sampled exploratory subset** of validation images, not an unbiased population-wide explanation study. It is designed to help you check the pipeline before expanding to a preregistered larger validation sample. Do **not** use the locked test partition for developing the explanation protocol.

## Outputs

`outputs/ACIT4620_10_Explainability_Validation/` includes image overlays, signed rule-contribution figures, reconstruction checks, Grad-CAM stability tables, input-occlusion results, and experiment metadata. The corresponding `_Outputs.zip` is generated at the end.

## Scientific limitations for the final paper

- No claim of medical validity or causal lesion-region explanation from Grad-CAM alone.
- Blurring top-attribution regions can create altered inputs and can change logits for reasons other than faithful attribution.
- Fuzzy rule indices are **permutation-sensitive across training seeds**; do not treat Rule 3 in seed 42 as semantically the same as Rule 3 in seed 52.
- Validation is reused for choosing classifiers/checkpoints; this limits any inferential claim from these samples.
- No test data are used, and no new CNN/ANFIS training is performed.
- Grad-CAM explanation stability and image-level faithfulness are investigated using fixed target class, equal perturbation budgets, and random/bottom-region controls; these alone do not establish causality.

## GitHub

Commit the notebook, source, tests, small checkpoints, scaler, reference predictions, output tables/figures and backup ZIP under a *new* `experiments/e6_xai_gradcam/` folder. **Do not commit `data/raw/` original HAM10000 images or `.venv/`.** Review `git diff --cached --name-only` before committing. See `PUSH_TO_GITHUB.md` after running the notebook.

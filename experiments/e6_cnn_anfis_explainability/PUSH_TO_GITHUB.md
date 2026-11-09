# Commit Notebook 10 to GitHub after execution

Use a fresh clone or a clean branch from `master`, e.g. `feature/e6-an​​fis-aware-gradcam`. Copy this project to `experiments/e6_xai_gradcam/`, excluding `.venv`, `__pycache__`, `.git`, and **`data/raw/`**. The included checkpoint files are each under GitHub's 100 MB limit. A sample PowerShell workflow from the clone root:

```powershell
$source = 'C:\path\to\ACIT4620_10_CNN_ANFIS_XAI_VSCode'
$destination = 'experiments\e6_xai_gradcam'
robocopy $source $destination /E /XD .venv .git __pycache__ .ipynb_checkpoints .pytest_cache raw /XF *.pyc
if ($LASTEXITCODE -ge 8) { throw 'Copy failed' }
git add -- experiments/e6_xai_gradcam
# Force-add only approved checkpoint/output artifacts that the team repository might ignore.
git add -f -- experiments/e6_xai_gradcam/data/model_assets
git add -f -- experiments/e6_xai_gradcam/outputs
git diff --cached --name-only
# STOP if any original HAM10000 images, credentials, patient information, or environment files appear.
git commit -m 'feat(e6): add ANFIS-aware Grad-CAM and occlusion analysis'
git push -u origin feature/e6-anfis-aware-gradcam
```

The local model checkpoints and validation-image IDs are purpose-specific research artifacts. The original HAM10000 dataset must be obtained separately by anyone reproducing image-level Grad-CAM.

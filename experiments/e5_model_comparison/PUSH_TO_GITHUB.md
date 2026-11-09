# Push Notebook 09 to ACIT4620-Group1-Hybrid-AI

**Do this only after checking the notebook output.** Use a clean clone outside OneDrive if previous branch switches failed to remove folders.

```powershell
cd "$env:USERPROFILE\Projects"
git clone https://github.com/GretteThel/ACIT4620-Group1-Hybrid-AI.git ACIT4620_Group1_N09
cd ACIT4620_Group1_N09
git switch -c feature/e5-model-comparison
```

Now **copy the entire extracted `ACIT4620_09_Model_Comparison_VSCode` folder** into this cloned repository as `experiments\e5_model_comparison` using File Explorer. Make sure there is one layer, not `e5_model_comparison/ACIT4620_09_Model_Comparison_VSCode/...`.

```powershell
git status --short
git add -- experiments/e5_model_comparison
git diff --cached --stat
git commit -m "feat(e5): add paired validation comparison of CNN, MLP and ANFIS"
git push -u origin feature/e5-model-comparison
```

Create a PR comparing `feature/e5-model-comparison` into `master`. Avoid `git add .` from any unexpected directory. Source prediction CSVs have original file hashes documented in `inputs/input_provenance.json` (except a redacted local path in the E3 config). No locked test data are included.

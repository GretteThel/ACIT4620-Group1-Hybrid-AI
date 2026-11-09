# Import complete E4 experiment to Group 1 GitHub

Target remote: `https://github.com/GretteThel/ACIT4620-Group1-Hybrid-AI.git`

This E4 bundle contains **all 150 original E4 result files**, all nine trained checkpoints, nine `.npz` prediction sets and rule-explanation arrays, every per-run CSV/plot/JSON, **plus the untouched original E4 results ZIP**. It also includes the source notebook, Python scripts, requirements, tests, and an RQ1–RQ4 analysis.

## Recommended PowerShell commands (from the ALREADY CLONED GitHub repository)

```powershell
# Verify repo before acting; its root must end in ACIT4620-Group1-Hybrid-AI.
git rev-parse --show-toplevel
git status

# Start a new branch from the current master, not the earlier E3 branch.
git switch master
git pull --ff-only origin master
git switch -c feature/e4-first-order-sugeno-anfis

# Extract the downloaded bundle directly into the REPOSITORY ROOT.
$bundle = Join-Path $env:USERPROFILE 'Downloads\E4_GitHub_Complete_With_All_Outputs.zip'
Expand-Archive -LiteralPath $bundle -DestinationPath (Get-Location).Path -Force

# Verify presence of the experiment and every E4 saved output.
$e4 = 'experiments\e4_sugeno_anfis'
if (!(Test-Path "$e4nfis_core.py")) { throw 'E4 model code missing' }
if (!(Test-Path "$e4\outputs\ACIT4620_E4_ANFIS_Outputs.zip")) { throw 'E4 backup missing' }
$files = @(Get-ChildItem "$e4\outputs\ACIT4620_E4_ANFIS" -Recurse -File)
$checkpoints = @($files | Where-Object { $_.Name -eq 'best_anfis_head.pth' })
if ($files.Count -ne 150 -or $checkpoints.Count -ne 9) { throw "E4 output audit failed: $($files.Count) files, $($checkpoints.Count) checkpoints" }

# Stage ONLY this isolated E4 experiment. Repo-level .gitignore ignores *.pth, *.npz.
git add -- "$e4"
git add -f -- "$e4/outputs"

# Review every staged path and file-size summary BEFORE committing.
git diff --cached --stat
git diff --cached --name-only

git commit -m 'feat(e4): add Sugeno ANFIS implementation and complete outputs'
git push -u origin feature/e4-first-order-sugeno-anfis
```

Then create a pull request with **base `master`**, **compare `feature/e4-first-order-sugeno-anfis`**. Review Files Changed and ensure no local `data/` inputs, environments, raw HAM10000 images, or credentials were added. Do not run `git add .` in unrelated directories.

**Storage note:** Saving both the unzipped results and the backup ZIP intentionally duplicates ~12 MB of content in Git. This is acceptable for this small E4 run but not a long-term binary-artifact strategy. If the group plans many experiments, consider Git LFS or external release artifacts instead; don't retroactively rewrite the published history without coordinating.

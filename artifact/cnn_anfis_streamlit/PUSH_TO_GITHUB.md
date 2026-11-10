# Publish the tested application to GitHub

**Repository:** `https://github.com/GretteThel/ACIT4620-Group1-Hybrid-AI`

The GitHub account is not connected for direct writes in this chat. These steps run on **your own Windows computer** using your already-authorized Git credentials. The workflow uses a separate feature branch and does not push directly into the shared `master` branch.

1. Verify the application loads on the local Streamlit URL and the existing unit tests pass.
2. Verify the seven-class **validation-only** parity check against your original validation reference (the actual images are held on your PC).
3. In PowerShell, ensure Git works (`git --version`) and that your Git credentials allow pushing to your group repository.
4. Run `publish_to_github.ps1` from the app folder. It clones the repository if no clone exists, refuses a dirty worktree, makes a branch named `feature/cnn-anfis-streamlit-final`, copies **only allowlisted source files** to `applications/cnn_anfis_streamlit`, and commits and pushes that branch. The original CNN/ANFIS/scaler files are **not** copied or committed.
5. Review the pushed files on GitHub and open a pull request to `master` for group review.

Example:

```powershell
$base = 'C:\Users\grela\OneDrive\Desktop\OSLOMET\COMPUTATIONAL INTELLIGENCE\FOR FINAL REPORT'
$app = "$base\ARTIFACT\ACIT4620_CNN_ANFIS_App"
$repo = "$base\CODE\ACIT4620-Group1-Hybrid-AI"
Set-Location $app
powershell -ExecutionPolicy Bypass -File .\publish_to_github.ps1 -AppDir $app -RepoDir $repo
```

If the existing repo has uncommitted work, save/commit that work first. If you already have a branch with this name, the script deliberately refuses to reuse it; choose a fresh name or follow normal Git branch/update procedures. If the remote rejects the push for authorization/permissions, sign into GitHub in Git Credential Manager or request repository access. Do not paste tokens into ChatGPT.

Do not add `model_assets/*.pth`, `*.npz`, `data/raw`, `outputs/`, `.venv`, `.env` or patient images to GitHub. See `.gitignore`.

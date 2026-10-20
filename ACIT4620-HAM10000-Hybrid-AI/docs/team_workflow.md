# GitHub collaboration guide (suggested)

## Recommended access
1. Create a **Private** GitHub repository named `ACIT4620-HAM10000-Hybrid-AI`.
2. Add the whole contents of this directory.
3. Go to **Settings → Collaborators** (or **Collaborators and teams**) and invite your teammates using GitHub usernames.
4. Keep `main` as an integration branch; each person works on `feature/<short-task-name>` branches and opens pull requests.

## Suggested responsibilities (assign names yourselves)
- Data & EDA: audit and manifests, quality gates.
- Modeling: EfficientNet-B0 and training experiments.
- Fuzzy AI: membership functions and rule design.
- Evaluation: metrics and uncertainty/review comparisons.
- Report & reproducibility: paper, figures, experiment tracking, merge oversight.

Work may overlap; every experiment should record responsible person, dataset version, seed, config, validation results, and commit hash. Do not commit raw medical images, huge model weights, secrets, or patient-identifying data.

## Suggested issue backlog
- [ ] Complete full image and content-hash audit; document exceptions.
- [ ] Save verified seven-class manifests / class mapping in Kaggle output and record Kaggle version.
- [ ] Add reproducible EfficientNet-B0 training notebook/script.
- [ ] Test weighted loss and training-only augmentation experiments.
- [ ] Produce confusion matrix, class-wise recall/F1 and error slices.
- [ ] Implement and explain Mamdani rule base.
- [ ] Benchmark fuzzy review scores vs simple thresholds.
- [ ] Write report with dataset citation, limitations and reproducibility details.

## Add the repository to GitHub from your computer
After creating an empty GitHub repository, unzip the package, open a terminal in the extracted directory, and run:

```bash
git init
git add .
git commit -m "Initialize seven-class HAM10000 research project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ACIT4620-HAM10000-Hybrid-AI.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your actual GitHub username. This is a **template address**, not a published repository link.

Alternative: in the new GitHub repository select **Add file → Upload files** and upload the extracted project files/folders (do not upload the zip alone if you want browsing and Git history).

# ACIT4620 Group 1 — Notebook 11: Computational Efficiency (RQ4)

## Purpose and proposal alignment

Research question **RQ4:** *What additional training, inference, and memory costs are associated with the neuro-fuzzy classifier, and how does the number of fuzzy rules influence the trade-off between predictive performance and computational efficiency?*

The notebook benchmarks six **selected model configurations** end to end and 16 saved model heads (CNN head + six E3 heads + nine E4 ANFIS heads) in head-only mode:

- E2a-50 EfficientNet-B0 original CNN classifier
- E3 Linear (three seeds) and MLP (three seeds)
- E4 first-order Takagi–Sugeno ANFIS (4, 8 and 16 Gaussian rules; three seeds each)

It uses **the original saved weights** and the same frozen CNN. No retraining, test-set access, or new predictive metrics occur.

## Setup in Windows VS Code

1. Extract ZIP into a **short path** such as `C:\AI\NB11`. Open the inner `ACIT4620_11_Efficiency_VSCode` folder.
2. Use the working E3 `.venv` Python interpreter that ran Notebooks 08 and 10; select it as the Notebook 11 kernel.
3. From the inner folder, install requirements using that interpreter: `python -m pip install -r requirements.txt`.
4. **E3 checkpoint input:** The six original E3 head checkpoints (and training-only scaler) must exist in a local clone of the team repository's merged `master` branch, e.g. `C:\AI\G1-NB10`. If automatic discovery fails, set `TEAM_REPO` in the notebook to the checkout root (the folder containing `experiments`). This package includes CNN and all nine ANFIS original checkpoint assets; it does **not** invent E3 weights.
5. **Real-image benchmark:** Set `RAW_IMAGES_ROOT` to the location of the `HAM10000_images_part_1` and `HAM10000_images_part_2` subfolders. The same `data/raw` directory of Notebook 10 is supported by automatic lookup. If raw images are unavailable, explicitly change `IMAGE_MODE = 'synthetic'`; label any resulting cost plots synthetic shape-matched, not real-image empirical timings.
6. Run cells sequentially. Outputs and the final backup ZIP are created under `outputs/`.

## Outputs

- `parameter_and_checkpoint_cost.csv` — head/total effective parameters and checkpoint sizes
- `historical_training_times_NOT_matched.csv` — original E3/E4 training durations **descriptive only**
- `training_step_microbenchmark.csv` — same-hardware per-step training cost using *synthetic* 1280-D features and simulated labels (not actual training)
- `head_only_inference.csv` — same-hardware head-only inference costs, all 16 models, batches 1/32/128
- `end_to_end_tensor_to_logit.csv` — CNN + head with identical preprocessed image tensor inputs; full network, excludes image decode and preprocessing
- `end_to_end_jpg_to_logit.csv` — optional real validation JPEG decode, ImageNet transform and network inference together
- `benchmark_provenance.json` — hardware/software, benchmark mode, source hashes and caveats
- `figures/` — ready-to-use comparison plots
- `ACIT4620_11_Efficiency_Results.zip` — complete output backup

## Interpretation boundaries

- **Historical training seconds are not directly comparable** to new benchmark measurements because they include different training histories, budgets, and potentially environments.
- **Microbenchmark training steps are not reported as completed training experiments.** They approximate the incremental compute cost of a forward, backward and optimizer update for a matched batch.
- **Head-only latency is NOT full inference latency.** Full models share CNN feature-extraction cost. Report both.
- CPU benchmark snapshots are process RSS before/after, **NOT isolated peak model memory**. Peak incremental GPU allocated memory is only available when run on CUDA.
- Timings depend on CPU/GPU, threads, batch size, warmup, OS contention and device precision. Avoid cross-device efficiency claims.
- Image selection uses only the pre-existing 50-image **validation** manifest from Notebook 10. Held-out test stays locked.

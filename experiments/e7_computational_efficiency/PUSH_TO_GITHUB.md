# Add Notebook 11 to GitHub after completing the real measurements

Target directory: `experiments/e7_computational_efficiency/`.

Copy this folder including original checkpoint assets, CSV outputs and figures. Do NOT copy a virtual environment or the raw HAM10000 images. Verify every staged path before committing.

Suggested PR title: `feat(e7): add matched computational efficiency benchmarks for CNN and ANFIS`

Describe the device, CPU thread count, benchmark repetition count, effective parameters, head-only and end-to-end latency, and memory caveats. Explain that original training time and synthetic training-step microbenchmark are distinct.

Do not report synthetic smoke-test numbers as empirical results. Do not access the held-out test set.

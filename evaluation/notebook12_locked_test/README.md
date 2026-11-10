# ACIT4620 - Notebook 12: Final Locked-Test Evaluation

This directory contains the final research-stage evaluation
implementation for the CNN, Linear, MLP, and Sugeno ANFIS models.

## Evaluation protocol

- Validation-selected model checkpoints are frozen before evaluation.
- Expected held-out test size: 1,458 images.
- No test-driven model selection or hyperparameter tuning.
- Seven-class metrics, confusion matrices, and paired lesion-level
  bootstrap comparisons are predefined.
- Three-seed checkpoint sensitivity is assessed separately.

## Reproduction

1. Install the dependencies.
2. Configure local paths in the notebook.
3. Restore the original trained checkpoint and scaler files.
4. Run unit tests.
5. Generate and verify the pre-test freeze.
6. Obtain genuine team approval before opening the locked test set.
7. Run the final evaluation once under the approved protocol.

The original local freeze, signed approval records, images, model weights,
and test outputs are not included here.

This evaluation is research-only and is not a clinical validation.

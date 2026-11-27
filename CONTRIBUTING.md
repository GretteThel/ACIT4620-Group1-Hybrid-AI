
# ACIT4620 Group 1 — Contribution Guidelines

## Project structure

The final implementation is contained in
`ACIT4620_Group1_Hybrid_AI.ipynb`.

The repository also includes selected experimental figures,
numerical result tables, configuration records, and
reproduction instructions in `README.md` and `results/`.

## Development and reproducibility

- Use the original HAM10000 dataset.
- Follow the Kaggle execution instructions in `README.md`.
- Preserve lesion-disjoint dataset partitioning.
- Fit preprocessing and standardization parameters
  using training data only.
- Perform model selection using training and development
  data, not the held-out test partition.
- Record configuration changes, random seeds, and
  experimental results.
- Preserve the original experiment outputs and provenance.

## Implementation verification

Before submitting changes:

1. Check that the notebook code cells contain valid Python.
2. Verify that the recorded configuration matches the
   reported experimental methodology.
3. Check the notebook's assertions and internal
   verification procedures.
4. Confirm that reported numerical results agree with
   the corresponding saved CSV files.
5. Review changes with the other group members.

## Evaluation integrity

The final experiment compares the CNN baseline,
first-order Sugeno ANFIS, and coverage-regularized
Sugeno ANFIS.

Do not modify the recorded held-out results or use them
to select alternative architectures or hyperparameters.

Any additional experiment must be clearly identified
and documented separately.

## Data and artifact management

Do not commit raw medical images, access tokens,
large model checkpoints, or unnecessary intermediate
training archives.

Preserve original experiment artifacts in approved
storage for reproducibility and examiner verification.

This project is a research prototype and is not a
validated clinical diagnostic system.

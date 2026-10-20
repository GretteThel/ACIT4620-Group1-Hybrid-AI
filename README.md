# ACIT4620 - Seven-class CNN + Mamdani fuzzy uncertainty review

This is a **research prototype**, not a diagnostic or medically validated triage system. CNN class labels remain unchanged when fuzzy review scores are computed.

## Files
- `fuzzy_review.py`: normalized entropy, deterministic TTA disagreement, Mamdani membership functions/rules/centroid, validation-based review budget, review metrics.
- `evaluate_review.py`: compares four review rankings at matched batch review budgets (exact top-k, no test labels used); also reports validation-derived cutoffs for a distinct fixed-threshold deployment experiment. Fuzzy ties are broken by prespecified entropy; measures outcomes on an untouched test set.
- `demo_synthetic.py`: self-contained demonstration using synthetic probabilities; does not train or assess a real CNN.

## Integration contract
Once EfficientNet-B0 is trained, **export calibrated CNN outputs** to two `.npz` files. Required arrays: `y_true` (int 0-6), `probs` (N,7), `tta_probs` (N,T,7). Include `lesion_id` strings (N,) where possible. Use exactly the class index order `nv, mel, bkl, bcc, akiec, vasc, df`. Use consistent deterministic TTA such as original, horizontal flip, vertical flip, 180-degree rotation. Predictions must come from the correct lesion-disjoint validation and test manifests. Fit calibration using validation data, freeze all modeling and fuzzy settings before first test evaluation. Never select fuzzy thresholds on the test labels.

```bash
python demo_synthetic.py
python evaluate_review.py --validation validation_predictions.npz --test test_predictions.npz --out results
```

Dependencies: `numpy`, `pandas`, `scikit-learn` (already standard in Kaggle).

Proposed membership boundaries are illustrative and not clinical. The fuzzy score means uncertainty-based review priority only; it is **not** a probability, disease severity, or medical treatment decision.

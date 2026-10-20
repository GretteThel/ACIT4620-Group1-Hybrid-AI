"""SMOKE TEST ONLY. Synthetic probabilities, not HAM10000/CNN performance."""
from pathlib import Path
import numpy as np
from fuzzy_review import batch_fuzzy_scores, scores_from_probs, review_metrics

rng = np.random.default_rng(42)
# Demonstrate review scores from illustrative seven-class probability arrays
probs = np.array([
    [.94, .01, .01, .01, .01, .01, .01], # confident prediction
    [.32, .29, .16, .09, .07, .04, .03], # uncertain prediction
    [.04, .81, .05, .03, .03, .02, .02], # confident melanoma prediction
    [.20, .18, .17, .13, .12, .11, .09], # diffuse classification
])
# Symmetric random perturbations for a reproducible nonclinical toy example
views = []
for p in probs:
    x = np.maximum(p + rng.normal(0, .02, (4,7)), .0001)
    x /= x.sum(axis=1,keepdims=True)
    views.append(x)
tta = np.stack(views)
h, d = scores_from_probs(probs, tta)
score = batch_fuzzy_scores(probs, tta)
for i in range(len(probs)):
    print(f'example {i}: entropy={h[i]:.3f}, TTA disagreement={d[i]:.3f}, fuzzy review={score[i]:.1f}/100')
assert np.isfinite(score).all() and ((score>=0)&(score<=100)).all()
print('Synthetic demo completed (not a trained-model evaluation).')

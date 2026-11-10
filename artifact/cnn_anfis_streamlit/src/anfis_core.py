"""Core first-order multi-output Takagi-Sugeno ANFIS.

Implements the architecture in the submitted ACIT4620 Group 1 proposal.
This is *not* a Mamdani post-processing or probability-correction layer.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans


class MultiOutputSugenoANFIS(nn.Module):
    """1280 -> 8 linear projection -> Gaussian rule premises -> seven affine logits.

    Equations (for projected x in R^8):
      mu[r,j] = exp(-0.5*((x_j - c[r,j])/sigma[r,j])**2)
      normalized_w[r] = softmax_r(sum_j log(mu[r,j]))
      consequent[r,c] = sum_j a[r,c,j] * x_j + b[r,c]
      logit[c] = sum_r normalized_w[r] * consequent[r,c]
    """
    def __init__(self, feature_dim=1280, latent_dim=8, n_rules=8, n_classes=7):
        super().__init__()
        if n_rules < 2 or latent_dim < 1:
            raise ValueError('At least two rules and a positive latent dimension are required')
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.n_rules = int(n_rules)
        self.n_classes = int(n_classes)
        self.projection = nn.Linear(feature_dim, latent_dim)
        self.centers = nn.Parameter(torch.zeros(n_rules, latent_dim))
        self.raw_widths = nn.Parameter(torch.zeros(n_rules, latent_dim))
        self.consequent_a = nn.Parameter(torch.empty(n_rules, n_classes, latent_dim))
        self.consequent_b = nn.Parameter(torch.zeros(n_rules, n_classes))
        nn.init.normal_(self.consequent_a, mean=0.0, std=0.02)

    def widths(self):
        # Explicit positivity, and a nonzero lower bound for stability.
        return 0.15 + F.softplus(self.raw_widths)

    def fuzzy_details(self, features):
        x = self.projection(features)  # explicitly *linear*, no activation
        sigma = self.widths()
        # log-domain product Gaussian firing. Softmax implements stable normalization.
        delta = (x[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :]
        log_firing = -0.5 * torch.sum(delta.square(), dim=-1)
        normalized_w = torch.softmax(log_firing, dim=1)
        consequents = torch.einsum('bd,rcd->brc', x, self.consequent_a) + self.consequent_b
        signed_contributions = normalized_w.unsqueeze(-1) * consequents
        logits = signed_contributions.sum(dim=1)
        return logits, x, normalized_w, consequents, signed_contributions, log_firing

    def forward(self, features):
        return self.fuzzy_details(features)[0]


@torch.no_grad()
def initialize_premises_train_only(model, x_train, device, seed):
    """Initialize fuzzy centers with KMeans on projected TRAINING features only.

    Unlike the subsequently learned projection and premises, KMeans here is only
    a deterministic train-only initialization—not a model-selection step.
    """
    model = model.to(device).eval()
    X = np.asarray(x_train, dtype=np.float32)
    projected = []
    for idx in range(0, len(X), 512):
        batch = torch.from_numpy(X[idx:idx + 512]).to(device)
        projected.append(model.projection(batch).cpu().numpy())
    z = np.concatenate(projected, axis=0)
    if not np.isfinite(z).all():
        raise ValueError('Non-finite projected training features')
    fitted = KMeans(n_clusters=model.n_rules, random_state=int(seed), n_init=5, max_iter=100).fit(z)
    # Widths reflect each projected feature's TRAINING distribution.
    widths = np.clip(z.std(axis=0, ddof=0) * 1.25, 0.35, 12.0).astype(np.float32)
    raw = np.log(np.expm1(np.maximum(widths - 0.15, 0.02))).astype(np.float32)
    model.centers.copy_(torch.from_numpy(fitted.cluster_centers_.astype(np.float32)).to(device))
    model.raw_widths.copy_(torch.from_numpy(np.tile(raw, (model.n_rules, 1))).to(device))
    return {'method': 'KMeans centers on frozen-CNN TRAIN projections only',
            'width_init': '1.25 times projected training standard deviation, clipped',
            'cluster_counts': np.bincount(fitted.labels_, minlength=model.n_rules).tolist()}


@torch.no_grad()
def rule_ablation_statistics(logits, weights, consequents, log_firing=None):
    """Validation-only faithfulness proxy for *selected predicted class*.

    For each example, remove the rule with the largest absolute signed
    contribution to the originally predicted logit. Renormalize surviving
    rule weights; compare with ablation of the smallest-contribution rule.
    Premises and consequents are held fixed (inference intervention only).
    """
    n, rules, classes = consequents.shape
    if rules < 2:
        raise ValueError('Rule ablation requires >=2 rules')
    assert logits.shape == (n, classes) and weights.shape == (n, rules)
    chosen = logits.argmax(dim=1)
    signed = weights * consequents.gather(2, chosen[:, None, None].expand(-1, rules, 1)).squeeze(-1)
    picks = {'max_abs_signed': signed.abs().argmax(dim=1),
             'min_abs_signed': signed.abs().argmin(dim=1)}
    base_probs = torch.softmax(logits, dim=1)
    base_chosen_p = base_probs.gather(1, chosen[:, None]).squeeze(1)
    summary = {}
    per_image = {'predicted_label': chosen.cpu().numpy()}
    for label, picked in picks.items():
        # Setting a rule to -infinity before Softmax faithfully removes the rule
        # from the product normalization, even if original weight ~1.0.
        masked = log_firing.clone() if log_firing is not None else torch.log(weights.clamp_min(1e-35))
        masked.scatter_(1, picked[:, None], float('-inf'))
        new_w = torch.softmax(masked, dim=1)
        new_logits = (new_w[:, :, None] * consequents).sum(dim=1)
        new_probs = torch.softmax(new_logits, dim=1)
        chosen_new_logit = new_logits.gather(1, chosen[:, None]).squeeze(1)
        chosen_old_logit = logits.gather(1, chosen[:, None]).squeeze(1)
        chosen_new_p = new_probs.gather(1, chosen[:, None]).squeeze(1)
        logit_change = (chosen_old_logit - chosen_new_logit).abs()
        prob_change = (base_chosen_p - chosen_new_p).abs()
        changed = (new_probs.argmax(dim=1) != chosen)
        summary[label] = {
            'mean_abs_original_predicted_logit_change': float(logit_change.mean().item()),
            'mean_abs_original_predicted_probability_change': float(prob_change.mean().item()),
            'fraction_predicted_class_changed': float(changed.float().mean().item()),
        }
        per_image[f'{label}_rule_index_zero_based'] = picked.cpu().numpy()
        per_image[f'{label}_abs_pred_logit_change'] = logit_change.cpu().numpy()
        per_image[f'{label}_abs_pred_probability_change'] = prob_change.cpu().numpy()
        per_image[f'{label}_pred_class_changed'] = changed.cpu().numpy()
    return summary, per_image

"""Run: python tests/test_anfis_core.py (does not access HAM10000)."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from anfis_core import MultiOutputSugenoANFIS, initialize_premises_train_only, rule_ablation_statistics


def test_shape_gradient_and_fuzzy_rules():
    torch.manual_seed(42)
    generator=np.random.default_rng(42)
    x=generator.normal(size=(80,1280)).astype('float32')
    y=torch.from_numpy(generator.integers(0,7,size=80).astype('int64'))
    for rules in [4,8,16]:
        model=MultiOutputSugenoANFIS(n_rules=rules)
        init=initialize_premises_train_only(model,x,torch.device('cpu'),42)
        assert len(init['cluster_counts']) == rules
        inputs=torch.from_numpy(x)
        logits,latent,w,consequents,signed,logw=model.fuzzy_details(inputs)
        assert logits.shape==(80,7)
        assert latent.shape==(80,8)
        assert w.shape==(80,rules)
        assert consequents.shape==(80,rules,7)
        assert signed.shape==(80,rules,7)
        assert torch.isfinite(logits).all()
        assert torch.allclose(w.sum(dim=1),torch.ones(80),atol=1e-6)
        assert torch.allclose(signed.sum(dim=1),logits,atol=1e-6)
        assert (model.widths()>0).all()
        loss=torch.nn.functional.cross_entropy(logits,y)
        loss.backward()
        for k in ['projection.weight','centers','raw_widths','consequent_a','consequent_b']:
            param=dict(model.named_parameters())[k]
            assert param.grad is not None and torch.isfinite(param.grad).all(),k
        summary,rows=rule_ablation_statistics(logits.detach(),w.detach(),consequents.detach(),logw.detach())
        assert summary['max_abs_signed']['mean_abs_original_predicted_logit_change']>=0
        assert len(rows['max_abs_signed_rule_index_zero_based']) == 80
        print('PASS',rules,'rules, parameters',sum(p.numel() for p in model.parameters()))

if __name__=='__main__':
    test_shape_gradient_and_fuzzy_rules()

"""Pure synthetic unit tests. Does NOT open original held-out test data."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from locked_test import (PRIMARY,RUNS,CLASSES,selection_table,performance,
                         classwise,cluster_bootstrap,canonical_sha,evaluate)

class Notebook12Tests(unittest.TestCase):
    def test_source_validation_selection(self):
        data=selection_table()
        self.assertEqual(data.model.tolist(),list(PRIMARY))
        self.assertEqual(len(RUNS),16)
    def test_seven_class_performance(self):
        y=np.arange(7).repeat(3)
        p=np.full((21,7),0.01)
        p[np.arange(21),y]=0.94
        result=performance(y,p)
        self.assertAlmostEqual(result['macro_f1'],1)
        self.assertAlmostEqual(result['melanoma_recall'],1)
        self.assertEqual(len(classwise(y,p,'synthetic')),7)
    def test_bootstrap_grouping(self):
        n=7*3
        data=pd.DataFrame({'lesion_id':[f'synthetic_{i}' for i in range(n)],
                           'label':np.arange(7).repeat(3)})
        p=np.full((n,7),0.0001,dtype=float)
        p[np.arange(n),data.label.to_numpy()]=0.9994
        predictions={x:p.copy() for x in PRIMARY}
        intervals=cluster_bootstrap(data,predictions,n_resamples=50,seed=11)
        diff=intervals.query("contrast=='anfis_r16_minus_mlp' and metric=='macro_f1'")
        self.assertEqual(len(diff),1)
        self.assertAlmostEqual(float(diff.interval_2p5.iloc[0]),0)
        self.assertAlmostEqual(float(diff.interval_97p5.iloc[0]),0)
    def test_test_stage_requires_explicit_consent(self):
        from argparse import Namespace
        with self.assertRaisesRegex(RuntimeError,'--confirm-locked-test'):
            evaluate(Namespace(confirm_locked_test=False))

    def test_lock_hash_tamper_detection(self):
        x={'source':'validation','selection':['cnn_e2a50']}
        first=canonical_sha(x)
        x['selection'].append('test_best_model')
        self.assertNotEqual(first,canonical_sha(x))

if __name__=='__main__':unittest.main()

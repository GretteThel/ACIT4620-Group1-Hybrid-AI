"""Lightweight architecture and cost-definition tests; no trained weights needed."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from efficiency_benchmark import LinearHead,MLPHead,FullPipeline,SELECTED,RUNS,CLASSES,SPLIT_SHA
from anfis_core import MultiOutputSugenoANFIS

class TestNotebook11(unittest.TestCase):
    def test_seven_classes_and_splits(self):
        self.assertEqual(CLASSES,['nv','mel','bkl','bcc','akiec','vasc','df'])
        self.assertEqual(len(SPLIT_SHA),64)
        self.assertEqual(len(RUNS),15)
        self.assertEqual(len(SELECTED),6)

    def test_original_e3_architectures(self):
        self.assertEqual(sum(p.numel() for p in LinearHead().parameters()),8967)
        self.assertEqual(sum(p.numel() for p in MLPHead().parameters()),10767)
        for model in [LinearHead(),MLPHead()]:
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.randn(3,1280)).shape),(3,7))

    def test_original_anfis_parameter_counts_and_gradients(self):
        for rules,expected in [(4,10564),(8,10880),(16,11512)]:
            m=MultiOutputSugenoANFIS(1280,8,rules,7)
            self.assertEqual(sum(p.numel() for p in m.parameters()),expected)
            x=torch.randn(2,1280)
            out=m(x)
            self.assertEqual(tuple(out.shape),(2,7))
            out.sum().backward()
            self.assertIsNotNone(m.centers.grad)
            self.assertIsNotNone(m.projection.weight.grad)

if __name__=='__main__':unittest.main()

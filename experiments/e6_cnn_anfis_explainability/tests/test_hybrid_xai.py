"""Code-only smoke test with genuine frozen checkpoints and a random input.
It does NOT represent a real HAM10000 analysis and does not use test images.
"""
from pathlib import Path
from hybrid_xai import smoke_test

def test_model_integration_smoke():
    smoke_test(Path(__file__).resolve().parents[1])

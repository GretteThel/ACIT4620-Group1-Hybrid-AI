"""ACIT4620 Group 1 — inference package built from the original frozen models."""
from .inference import load_bundle, ClassifierBundle, CLASSES

__all__ = ["load_bundle", "ClassifierBundle", "CLASSES"]

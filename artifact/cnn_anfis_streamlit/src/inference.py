"""Frozen EfficientNet-B0 + 16-rule ANFIS inference (ACIT4620 Group 1).

Matches the original E2a-50/E4-ANFIS/Notebook10+12 inference graph:
- PIL EXIF transpose; RGB; shortest-side Resize(256); CenterCrop(224)
- ImageNet ToTensor + channel normalization
- EfficientNet-B0 features -> avgpool -> 1280-D features
- train-only mean/std feature standardization (NO refitting)
- 1280->8 linear projection -> 16 Gaussian first-order Sugeno rules -> seven logits.

Only images explicitly provided by caller are opened. Nothing here loads a
train/validation/test manifest or changes the frozen pre-test selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0

from .anfis_core import MultiOutputSugenoANFIS

CLASSES = ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
CLASS_DESCRIPTIONS = {
    "nv": "Melanocytic nevus",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesion",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "vasc": "Vascular lesion",
    "df": "Dermatofibroma",
}
SPLIT_SHA256 = "265ec2412f4dd7d50c75b46d75a095c0d3445250895b90b91f3d73aca9aebce0"
CHECKSUMS = {
    "e2a50_best_efficientnet_b0.pth": "2efbe67904bed16aec8fbf1f47e3af11f1a0136e7eab4a4c5cbd004edf9b9af9",
    "anfis_r16_seed52.pth": "020663d064b6e7cb5b12b70d84974dd652c7d78c3759d1f696ade08de786dc18",
    "train_feature_scaler.npz": "290c5e6b418bdfcb8fe5436d313cfecdc8661f5fbc5447a4529b2e6d889fb43f",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_assets(assets_root: str | Path) -> dict[str, str]:
    root = Path(assets_root).expanduser().resolve()
    actual = {}
    for filename, expected in CHECKSUMS.items():
        f = root / filename
        if not f.is_file():
            raise FileNotFoundError(f"Missing original frozen model asset: {f}")
        current = sha256(f)
        if current != expected:
            raise ValueError(
                f"STOP: SHA-256 mismatch for {filename}; original frozen weights/scaler changed. "
                f"Expected {expected}; got {current}. Do not retrain or substitute a different file."
            )
        actual[filename] = current
    return actual


def _torch_checkpoint(path: Path) -> dict[str, Any]:
    # Only use SHA-256-verified local training artifacts, not arbitrary downloaded weights.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid checkpoint container: {path}")
    if data.get("split_sha256") != SPLIT_SHA256 or tuple(data.get("class_codes", [])) != CLASSES:
        raise ValueError(f"Original split/class-order provenance mismatch: {path}")
    return data


def prepare_image(image: Image.Image) -> torch.Tensor:
    """Exact CNN validation/locked-test transform; does not fit any preprocessing."""
    corrected = ImageOps.exif_transpose(image).convert("RGB")
    return TRANSFORM(corrected).unsqueeze(0)


def normalized_rgb_from_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """Return 224x224 model-view RGB for accurate Grad-CAM overlay alignment."""
    arr = image_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    arr = arr * np.asarray(IMAGENET_STD).reshape(1, 1, 3) + np.asarray(IMAGENET_MEAN).reshape(1, 1, 3)
    return np.clip(arr, 0, 1).astype(np.float32)


@dataclass
class ClassifierBundle:
    cnn: nn.Module
    anfis: MultiOutputSugenoANFIS
    scaler_mean: torch.Tensor
    scaler_std: torch.Tensor
    device: torch.device
    checksums: dict[str, str]

    def _tensors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = self.cnn.features(x)
        pooled = torch.flatten(self.cnn.avgpool(spatial), 1)
        normalized = (pooled - self.scaler_mean) / self.scaler_std
        return spatial, pooled, normalized

    @torch.no_grad()
    def predict_tensor(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        """Evaluation-mode predictions; accepts an already preprocessed 1x3x224x224 tensor."""
        if tuple(image_tensor.shape) != (1, 3, 224, 224):
            raise ValueError("Expected one preprocessed image tensor of shape (1,3,224,224)")
        x = image_tensor.to(self.device)
        _, pooled, normalized = self._tensors(x)
        cnn_logits = self.cnn.classifier(pooled)
        anfis_logits, projected, weights, consequent, signed, _ = self.anfis.fuzzy_details(normalized)
        cnn_prob = torch.softmax(cnn_logits, dim=1)[0].detach().cpu().numpy()
        anfis_prob = torch.softmax(anfis_logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(anfis_prob))
        if not (np.isfinite(cnn_prob).all() and np.isfinite(anfis_prob).all()):
            raise RuntimeError("Nonfinite prediction probabilities")
        if not (np.isclose(cnn_prob.sum(),1.,atol=1e-5) and np.isclose(anfis_prob.sum(),1.,atol=1e-5)):
            raise RuntimeError("Prediction probabilities do not sum to one")
        w = weights[0].detach().cpu().numpy()
        contrib = signed[0, :, pred].detach().cpu().numpy()
        c = consequent[0, :, pred].detach().cpu().numpy()
        sorted_by_activation = np.argsort(-w).tolist()
        sorted_by_abs_contrib = np.argsort(-np.abs(contrib)).tolist()
        return {
            "predicted_class": CLASSES[pred],
            "predicted_class_description": CLASS_DESCRIPTIONS[CLASSES[pred]],
            "predicted_class_index": pred,
            "predicted_probability": float(anfis_prob[pred]),
            "cnn_predicted_class": CLASSES[int(np.argmax(cnn_prob))],
            "classes": list(CLASSES),
            "anfis_probabilities": {k:float(v) for k,v in zip(CLASSES,anfis_prob)},
            "cnn_probabilities": {k:float(v) for k,v in zip(CLASSES,cnn_prob)},
            "fuzzy": {
                "n_rules": 16,
                "latent_projection": projected[0].detach().cpu().tolist(),
                "normalized_rule_firing": w.astype(float).tolist(),
                "signed_rule_contribution_to_predicted_logit": contrib.astype(float).tolist(),
                "rule_consequents_for_predicted_class": c.astype(float).tolist(),
                "highest_firing_rule_index_1_based": sorted_by_activation[0]+1,
                "highest_absolute_logit_contribution_rule_index_1_based": sorted_by_abs_contrib[0]+1,
                "predicted_logit_reconstruction": float(contrib.sum()),
            },
            "model": "EfficientNet-B0 frozen features + 16-rule Sugeno ANFIS (seed 52)",
            "research_use_only": True,
        }

    def predict_pil(self, image: Image.Image) -> dict[str, Any]:
        return self.predict_tensor(prepare_image(image))

    def predict_path(self, path: str | Path) -> dict[str, Any]:
        with Image.open(path) as img:
            return self.predict_pil(img)

    def gradcam(self, image_tensor: torch.Tensor, class_index: int, head: str = "anfis") -> np.ndarray:
        """Logit-based Grad-CAM on EfficientNet last spatial map (Notebook10 method)."""
        if head not in ("anfis", "cnn"):
            raise ValueError("head must be 'anfis' or 'cnn'")
        if not 0 <= int(class_index) < 7:
            raise ValueError("class index must be between 0 and 6")
        if tuple(image_tensor.shape) != (1,3,224,224):
            raise ValueError("Expected preprocessed tensor shape (1,3,224,224)")
        with torch.enable_grad():
            x = image_tensor.detach().clone().to(self.device).requires_grad_(True)
            spatial, pooled, norm = self._tensors(x)
            logits = self.anfis(norm) if head == "anfis" else self.cnn.classifier(pooled)
            grads = torch.autograd.grad(logits[0, int(class_index)], spatial)[0]
            alpha = grads.mean(dim=(2,3), keepdim=True)
            heat = (spatial * alpha).sum(dim=1, keepdim=True).clamp_min(0)
            up = F.interpolate(heat,size=x.shape[-2:], mode="bilinear", align_corners=False)
            arr = up.detach().cpu().float().numpy()[0,0]
            lo,hi = float(arr.min()),float(arr.max())
            return ((arr-lo)/(hi-lo+1e-8)).astype(np.float32)

    def make_overlay(self, image_tensor: torch.Tensor, heatmap: np.ndarray, alpha: float=0.47) -> Image.Image:
        """Overlay normalized CAM on the *center-cropped model-view* of the input."""
        import matplotlib
        if heatmap.shape != (224,224):
            raise ValueError("Heatmap must be 224x224")
        if not 0. <= alpha <= 1.:
            raise ValueError("alpha must be in [0,1]")
        rgb = normalized_rgb_from_tensor(image_tensor)
        colored = matplotlib.colormaps["magma"](heatmap)[...,:3]
        merged = np.clip((1-alpha)*rgb + alpha*colored, 0, 1)
        return Image.fromarray(np.uint8(np.round(merged*255)), mode="RGB")


def load_bundle(assets_root: str | Path, device: str = "cpu") -> ClassifierBundle:
    """Verify exact frozen original assets before loading; never downloads/retrains."""
    if device not in ("cpu", "cuda"):
        raise ValueError("Supported devices: 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    device_obj = torch.device(device)
    base = Path(assets_root).expanduser().resolve()
    digests = validate_assets(base)
    cnn_ck = _torch_checkpoint(base / "e2a50_best_efficientnet_b0.pth")
    anf_ck = _torch_checkpoint(base / "anfis_r16_seed52.pth")
    if int(anf_ck.get("n_rules",-1)) != 16 or int(anf_ck.get("seed",-1)) != 52:
        raise ValueError("ANFIS rule count/seed mismatch: expected 16/52")
    if anf_ck.get("source_checkpoint_sha256") != CHECKSUMS["e2a50_best_efficientnet_b0.pth"]:
        raise ValueError("ANFIS feature-extractor provenance mismatch")

    cnn = efficientnet_b0(weights=None)
    if cnn.classifier[1].in_features != 1280:
        raise ValueError("Unexpected EfficientNet-B0 pooled feature size")
    cnn.classifier[1] = nn.Linear(1280, len(CLASSES))
    cnn.load_state_dict(cnn_ck["model_state"], strict=True)
    cnn.eval().requires_grad_(False).to(device_obj)

    anfis = MultiOutputSugenoANFIS(1280,8,16,7)
    anfis.load_state_dict(anf_ck["state_dict"],strict=True)
    anfis.eval().requires_grad_(False).to(device_obj)

    with np.load(base / "train_feature_scaler.npz",allow_pickle=False) as z:
        mean=np.asarray(z["mean"],dtype=np.float32)
        std=np.asarray(z["std"],dtype=np.float32)
    if (mean.shape != (1280,) or std.shape != (1280,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std>0).all()):
        raise ValueError("Original training-only feature scaler has invalid data")
    mean_tensor = torch.from_numpy(mean.copy()).to(device_obj).view(1,-1)
    std_tensor = torch.from_numpy(std.copy()).to(device_obj).view(1,-1)
    return ClassifierBundle(cnn,anfis,mean_tensor,std_tensor,device_obj,digests)

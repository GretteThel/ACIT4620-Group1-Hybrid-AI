"""Small testable, non-clinical helpers for ACIT4620 Streamlit research UI.

No uploads are saved by this module. Never treat a softmax score as risk.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from typing import Any

from PIL import Image, UnidentifiedImageError

# Keep uploads bounded in local Streamlit demos; Streamlit's own uploader limit
# is also configured separately in .streamlit/config.toml.
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG"})


def read_demo_image(blob: bytes) -> Image.Image:
    """Decode user-selected JPEG/PNG safely and return an in-memory image.

    Image-specific provenance is not inferred; operator must use known
    validation or external demo images, never the sealed test partition.
    """
    if not isinstance(blob, bytes) or not blob:
        raise ValueError("Select a non-empty JPEG or PNG image.")
    if len(blob) > MAX_FILE_BYTES:
        raise ValueError("Image is too large for this demo (maximum 12 MiB).")
    try:
        with Image.open(BytesIO(blob)) as opened:
            if opened.format not in SUPPORTED_FORMATS:
                raise ValueError("Only JPEG or PNG images are accepted.")
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the demo's 16-megapixel limit.")
            # Load while BytesIO remains alive. Retain original orientation metadata;
            # src.inference.prepare_image applies the original EXIF transposition.
            opened.load()
            return opened.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not read this JPEG or PNG file.") from exc


def image_digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def prediction_table_rows(prediction: dict[str, Any], descriptions: dict[str, str]) -> list[dict[str, Any]]:
    """Preserve exact original seven-class order from the inference engine."""
    classes = prediction["classes"]
    if len(classes) != 7 or len(set(classes)) != 7:
        raise ValueError("Invalid seven-class prediction record.")
    return [
        {
            "Class": descriptions[code],
            "Code": code,
            "CNN–ANFIS": 100 * float(prediction["anfis_probabilities"][code]),
            "CNN only": 100 * float(prediction["cnn_probabilities"][code]),
        }
        for code in classes
    ]


def rule_table_rows(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Rule weights are dimensionless; signed contributions are to logits."""
    fuzzy = prediction["fuzzy"]
    weights = fuzzy["normalized_rule_firing"]
    signed = fuzzy["signed_rule_contribution_to_predicted_logit"]
    if len(weights) != 16 or len(signed) != 16:
        raise ValueError("Expected 16 fuzzy rules.")
    return [
        {
            "Rule": f"Rule {i + 1}",
            "Firing strength": float(weight),
            "Predicted-logit contribution": float(contribution),
        }
        for i, (weight, contribution) in enumerate(zip(weights, signed))
    ]


def export_json(prediction: dict[str, Any], sha256_digest: str, verified_assets: dict[str, str]) -> str:
    """Export only predictions and audit hashes, never uploaded image pixels."""
    record = {
        "intended_use": "ACIT4620 non-clinical research demonstration",
        "uploaded_image_sha256": sha256_digest,
        "original_model_asset_sha256": verified_assets,
        "prediction": prediction,
        "limitations": [
            "Not validated for clinical diagnosis or triage.",
            "Softmax scores are uncalibrated and not cancer-risk estimates.",
            "Grad-CAM is a model attribution view, not segmentation or causal evidence.",
            "Fuzzy rule contributions are signed effects on logits, not class probabilities.",
        ],
    }
    return json.dumps(record, indent=2, ensure_ascii=False)

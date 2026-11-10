"""Dependency-light, original-model-independent tests for UI helpers."""
import io
import json
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app_utils import (
    MAX_FILE_BYTES, export_json, image_digest, prediction_table_rows,
    read_demo_image, rule_table_rows,
)


class AppUtilitiesTests(unittest.TestCase):
    def sample_image(self, fmt="PNG"):
        b = io.BytesIO()
        Image.new("RGB", (240, 235), (153, 89, 80)).save(b, format=fmt)
        return b.getvalue()

    def test_png_and_jpeg_decode(self):
        for fmt in ("PNG", "JPEG"):
            im = read_demo_image(self.sample_image(fmt))
            self.assertEqual(im.size, (240, 235))
            self.assertEqual(im.mode, "RGB")

    def test_invalid_and_oversized_upload_rejected(self):
        for b in (b"", b"not a picture", b"x" * (MAX_FILE_BYTES + 1)):
            with self.assertRaises(ValueError):
                read_demo_image(b)

    def test_giant_image_is_rejected_before_decoding_pixels(self):
        # A tiny-file PNG with huge dimensions; image header guard is pre-decode.
        b = io.BytesIO()
        Image.new("RGB", (5000, 4000)).save(b, format="PNG", compress_level=9)
        with self.assertRaises(ValueError):
            read_demo_image(b.getvalue())

    def test_digest(self):
        self.assertEqual(
            image_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_class_order_and_fuzzy_semantics(self):
        classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
        dummy = {
            "classes": classes,
            "anfis_probabilities": {c: 1 / 7 for c in classes},
            "cnn_probabilities": {c: 1 / 7 for c in classes},
            "fuzzy": {
                "normalized_rule_firing": [1 / 16] * 16,
                "signed_rule_contribution_to_predicted_logit": [-0.5] * 16,
            },
        }
        rows = prediction_table_rows(dummy, {c: c for c in classes})
        self.assertEqual([r["Code"] for r in rows], classes)
        self.assertAlmostEqual(rows[0]["CNN–ANFIS"], 100 / 7)
        rule_rows = rule_table_rows(dummy)
        self.assertEqual(len(rule_rows), 16)
        self.assertEqual(rule_rows[0]["Predicted-logit contribution"], -0.5)
        payload = json.loads(export_json(dummy, image_digest(b"abc"), {"a": "hash"}))
        self.assertEqual(payload["original_model_asset_sha256"], {"a": "hash"})
        self.assertNotIn("uploaded_image_data", payload)


if __name__ == "__main__":
    unittest.main()

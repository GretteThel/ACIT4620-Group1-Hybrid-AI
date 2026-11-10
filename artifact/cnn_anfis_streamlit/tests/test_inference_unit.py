"""No test-image access; synthetic unit tests, plus optional real pinned checkpoint checks."""
import sys
import unittest
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.inference import CLASSES, prepare_image, validate_assets, sha256, CHECKSUMS

class InferenceUnitTests(unittest.TestCase):
    def test_class_order(self):
        self.assertEqual(CLASSES,('nv','mel','bkl','bcc','akiec','vasc','df'))

    def test_input_tensor_shape_and_finiteness(self):
        rgb=Image.new('RGB',(400,275), color=(183,112,90))
        x=prepare_image(rgb)
        self.assertEqual(tuple(x.shape),(1,3,224,224))
        self.assertTrue(np.isfinite(x.numpy()).all())

    def test_grayscale_image_promoted_to_rgb(self):
        x=prepare_image(Image.new('L',(230,240), color=128))
        self.assertEqual(tuple(x.shape),(1,3,224,224))

    def test_corrupt_or_missing_asset_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                validate_assets(d)
            for filename in CHECKSUMS:
                Path(d,filename).write_bytes(b'NOT A REAL CHECKPOINT')
            with self.assertRaises(ValueError):
                validate_assets(d)

    def test_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'abc')
            self.assertEqual(sha256(p),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

if __name__=='__main__':unittest.main()

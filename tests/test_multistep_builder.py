"""Atomic PNG freezing and preservation of already frozen bank files."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin
from scripts import build_multistep_cases as builder


def checksum(data):
    return hashlib.sha256(data).hexdigest()


class CanonicalizationTests(unittest.TestCase):
    def fixture(self, root):
        image = Image.new("RGB", (12, 10), (91, 52, 23))
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("File", "private-render.png")
        path = root / "image.png"
        image.save(path, pnginfo=metadata)
        row = {"path": path.name, "size": list(image.size),
               "sha256": checksum(path.read_bytes()),
               "decoded_rgb_sha256": checksum(image.tobytes())}
        raw = {"images": {"reference": row}, "seed": 7}
        manifest = root / "bank.json"
        manifest.write_text(json.dumps(raw, indent=2) + "\n")
        expected = BytesIO()
        image.save(expected, format="PNG")
        return path, manifest, raw, expected.getvalue()

    def snapshot(self, root):
        return {str(p.relative_to(root)): (checksum(p.read_bytes()), p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}

    def test_canonical_bytes_then_complete_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, raw, expected = self.fixture(root)
            original = raw["images"]["reference"]["sha256"]
            path.chmod(0o640)
            builder.canonicalize(manifest, raw)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual(raw["images"]["reference"]["renderer_file_sha256"], original)
            with Image.open(path) as image:
                self.assertFalse(image.info)
            before = self.snapshot(root)
            builder.canonicalize(manifest, raw)
            self.assertEqual(self.snapshot(root), before)
            self.assertEqual(json.loads(manifest.read_text()), raw)

    def test_corrupt_input_rejected_without_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, raw, _ = self.fixture(root)
            path.write_bytes(b"corrupt")
            before, original = self.snapshot(root), deepcopy(raw)
            with self.assertRaises(ValueError):
                builder.canonicalize(manifest, raw)
            self.assertEqual(self.snapshot(root), before)
            self.assertEqual(raw, original)

    def test_all_inputs_checked_before_first_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, manifest, raw, _ = self.fixture(root)
            second = root / "second.png"
            second.write_bytes(b"corrupt")
            raw["images"]["second"] = {**raw["images"]["reference"], "path": second.name}
            before, original = self.snapshot(root), deepcopy(raw)
            with self.assertRaises(ValueError):
                builder.canonicalize(manifest, raw)
            self.assertEqual(self.snapshot(root), before)
            self.assertEqual(raw, original)

    def test_failed_replace_keeps_original_and_cleans_own_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, raw, _ = self.fixture(root)
            before, original = self.snapshot(root), deepcopy(raw)
            with patch.object(builder.os, "replace", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    builder.canonicalize(manifest, raw)
            self.assertEqual(self.snapshot(root), before)
            self.assertEqual(raw, original)


if __name__ == "__main__":
    unittest.main()

"""Condition targets cite actual photographs without exposing source defect masks."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from resolveai.conversation import Conversation
from resolveai.environment import Environment
from scripts.build_condition_sft import make_case


class ConditionSFTTests(unittest.TestCase):
    def test_source_mask_is_private_and_reference_is_a_different_item(self):
        class NoOCR:
            available = False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("target.png", "reference.png"):
                Image.new("RGB", (40, 30), "gray").save(root / name)
            for label, expected in (("normal", "Refuted"), ("anomaly", "Supported")):
                case, call, paths = make_case(
                    {"image": "target.png", "label": label, "mask": "private-mask.png",
                     "anomaly_bbox": [5, 5, 10, 10]}, "reference.png", root)
                conversation = Conversation(Environment(case, root, ocr_backend=NoOCR()))
                public = json.dumps(conversation.public())
                self.assertNotIn("private-mask", public)
                self.assertNotIn("target_bbox", public)
                self.assertIn("another physical item", public)
                self.assertNotIn("source-derived-unreviewed", public)
                self.assertEqual(call["arguments"]["verdict"], expected)
                self.assertEqual(call["arguments"]["citations"][0]["bbox"], [0, 0, 40, 30])
                self.assertEqual(len(paths), 2)
                conversation.call(call)
                self.assertEqual(conversation.errors, 0)
                self.assertEqual(conversation.termination, "finished")


if __name__ == "__main__":
    unittest.main()

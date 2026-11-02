"""Check data isolation and the lack of hidden access-version hints in training."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from scripts.build_acquisition_sft import assert_disjoint, make_case, split_categories


class AcquisitionDataTests(unittest.TestCase):
    def test_rejects_eval_groups_and_pixel_duplicates(self):
        groups = {"train": {"co3d:a"}, "validation": {"co3d:b"}}
        hashes = {"train": {"sha-a"}, "validation": {"sha-b"}}
        assert_disjoint(groups, hashes, {"co3d:test"}, {"test-pixel"})
        with self.assertRaises(ValueError):
            assert_disjoint(groups, hashes, {"co3d:a"}, set())
        with self.assertRaises(ValueError):
            assert_disjoint(groups, hashes, set(), {"sha-b"})
        bad = deepcopy(hashes)
        bad["validation"].add("sha-a")
        with self.assertRaises(ValueError):
            assert_disjoint(groups, bad, set(), set())

    def test_unreleased_slot_and_request_have_no_access_version_hint(self):
        from PIL import Image
        from resolveai.conversation import Conversation
        from resolveai.environment import Environment
        class NoOCR:
            available = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for name in ("a", "b"):
                path = root / (name + ".png")
                Image.new("RGB", (32, 32), (20, 50, 80)).save(path)
                rows.append({"path": str(path), "bbox": [2, 2, 30, 30]})
            prompts, outcomes = [], []
            for variant in ("Obtainable", "Missing", "Unavailable"):
                case, final, _ = make_case(*rows, True, variant, root)
                convo = Conversation(Environment(case, root, ocr_backend=NoOCR()))
                prompts.append(convo.public())
                initial_images = [i for m in convo.messages if isinstance(m.get("content"), list)
                                  for i in m["content"] if i.get("type") == "image"]
                self.assertEqual(len(initial_images), 1)
                response = convo.call({"name": "request_photo", "arguments": {
                    "query": {"object": "subject-B", "time": "capture", "view": "view-02"}}})
                outcomes.append(response["status"])
                convo.call(final)
                self.assertEqual(convo.errors, 0)
                self.assertEqual(convo.termination, "finished")
            self.assertEqual(prompts[0], prompts[1])
            self.assertEqual(prompts[1], prompts[2])
            self.assertEqual(outcomes, ["provided", "unable_to_provide", "unable_to_provide"])

    def test_category_split_is_complete_disjoint_and_repeatable(self):
        categories = [str(i) for i in range(20)]
        partition = split_categories(categories)
        self.assertEqual(partition, split_categories(reversed(categories)))
        self.assertFalse(set(partition["train"]) & set(partition["validation"]))
        self.assertEqual(set(categories), set(partition["train"] + partition["validation"]))


if __name__ == "__main__":
    unittest.main()

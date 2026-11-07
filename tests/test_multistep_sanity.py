"""Synthetic RGB unit tests for the renderer-specific sanity readout."""
import io
import unittest

import cv2
import numpy as np
from PIL import Image

from scripts import evaluate_multistep_sanity as sanity


def public_image(name, array):
    stream = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(stream, format="PNG")
    height, width = array.shape[:2]
    return {"image_id": name, "image_png": stream.getvalue(),
            "source_bbox": [0, 0, width, height], "time": "inspection"}


def pattern(seed):
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(rng.normal(0, 12, (100, 80)), (0, 0), 2)
    field *= 10
    return np.clip(np.array([143, 100, 60]) + field[:, :, None] *
                   np.array([1, .5, .25]), 30, 200).astype(np.uint8)


def scene(order):
    reference = np.full((160, 160, 3), 220, dtype=np.uint8)
    reference[30:130, 40:120] = pattern(0)
    front = np.full((160, 380, 3), 220, dtype=np.uint8)
    for i, identity in enumerate(order):
        front[30:130, 20 + 120*i:100 + 120*i] = pattern(identity)
    return {"images": [public_image("reference", reference),
                       public_image("front_survey", front)]}


class SanityRGBTests(unittest.TestCase):
    def test_request_target_depends_on_visible_texture(self):
        for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
            result = sanity.match_front(scene(order))
            self.assertIsNotNone(result)
            self.assertEqual(result["slot"], sanity.SLOTS[order.index(0)])

    def test_degenerate_textures_abstain(self):
        observation = scene((0, 1, 2))
        for image in observation["images"]:
            array = sanity.pixels(image).copy()
            brown = np.max(array, axis=2) < 210
            array[brown] = [143, 100, 60]
            image.update(public_image(image["image_id"], array))
        self.assertIsNone(sanity.match_front(observation))

    def test_bright_scratch_bbox_from_pixels(self):
        array = np.full((160, 160, 3), 220, dtype=np.uint8)
        array[30:130, 40:120] = [143, 100, 60]
        intact = sanity.diagnose(public_image("rear_left", array))
        self.assertEqual(intact["verdict"], "Refuted")
        cv2.line(array, (60, 60), (100, 100), (177, 155, 122), 2)
        scratched = sanity.diagnose(public_image("rear_left", array))
        self.assertEqual(scratched["verdict"], "Supported")
        x0, y0, x1, y1 = scratched["diagnostic"]["bbox"]
        self.assertLessEqual(x0, 60)
        self.assertLessEqual(y0, 60)
        self.assertGreater(x1, 100)
        self.assertGreater(y1, 100)

    def test_no_target_rear_never_invents_evidence(self):
        observation = scene((2, 0, 1))
        decision, _ = sanity.readout(observation)
        self.assertEqual(decision["verdict"], "Need more evidence")
        self.assertEqual(decision["citations"], [])
        self.assertEqual(decision["links"], [])

    def test_nonimage_metadata_cannot_supply_identity(self):
        observation = scene((2, 0, 1))
        before = sanity.match_front(observation)
        observation["private_state"] = {"target": "right", "condition": "scratched"}
        observation["annotation"] = {"verdict": "Refuted"}
        self.assertEqual(sanity.match_front(observation), before)


if __name__ == "__main__":
    unittest.main()

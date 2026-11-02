"""Regression checks for native processor settings and Pixtral system messages."""
import json
from pathlib import Path
import tempfile
import unittest

from resolveai.model_processing import processor_options, typed_messages, chat_inputs


class ModelProcessingTests(unittest.TestCase):
    def test_image_budget_and_unknown_loader_options_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolveai_processor.json"
            self.assertEqual(processor_options(directory), {})
            for value in ({"size": {"longest_edge": 1024}}, {"trust_remote_code": True},
                          {"fix_mistral_regex": "yes"}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    processor_options(directory)
            value = {"size": {"longest_edge": 512}, "fix_mistral_regex": True}
            path.write_text(json.dumps(value))
            self.assertEqual(processor_options(directory), value)

    def test_pixtral_system_string_is_preserved_without_changing_input(self):
        messages = [{"role": "system", "content": "Only use public evidence."},
                    {"role": "user", "content": [{"type": "text", "text": "Compare images."}]}]
        result = typed_messages(messages, "PixtralProcessor")
        self.assertEqual(result[0]["content"], messages[0]["content"])
        self.assertIsNot(result[1]["content"], messages[1]["content"])
        self.assertIsInstance(typed_messages(messages, "Qwen3VLProcessor")[0]["content"], list)

    def test_pixtral_uses_render_only_and_ordered_pixels(self):
        class PixtralProcessor:
            def apply_chat_template(self, messages, *, tokenize, **kwargs):
                self_test.assertFalse(tokenize)
                self_test.assertEqual(messages[0]["content"], "Instructions")
                return "rendered checkpoint prompt"
            def __call__(self, *, text, images, return_tensors):
                return {"text": text, "images": images}
        self_test = self
        messages = [{"role": "system", "content": "Instructions"},
                    {"role": "user", "content": [
                        {"type": "image", "image": "first"},
                        {"type": "text", "text": "then"},
                        {"type": "image", "image": "second"}]}]
        result = chat_inputs(PixtralProcessor(), messages)
        self.assertEqual(result["images"], ["first", "second"])
        self.assertEqual(result["text"], "rendered checkpoint prompt")

    def test_pixtral_existing_system_blocks_preserve_all_text(self):
        messages = [{"role": "system", "content": [
            {"type": "text", "text": "First. "}, {"type": "text", "text": "Second."}]}]
        self.assertEqual(typed_messages(messages, "PixtralProcessor")[0]["content"], "First. Second.")
        messages[0]["content"].append({"type": "image", "image_path": "private.png"})
        with self.assertRaises(ValueError):
            typed_messages(messages, "PixtralProcessor")


if __name__ == "__main__":
    unittest.main()

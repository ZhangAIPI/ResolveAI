"""Interface failures must remain distinguishable from model abstention."""
import unittest
from scripts.check_model_interfaces import verdict_from_text
from resolveai.vlm_worker import parse_tool_call, portable_tools


class InterfaceTests(unittest.TestCase):
    def test_explicit_verdicts_and_invalid_answers(self):
        for answer in ["Supported", "Refuted", "Need more evidence"]:
            self.assertEqual(verdict_from_text("VERDICT: " + answer), answer)
        for answer in ["", "I cannot parse this", "Supported or Refuted", "C",
                       "Supported\nRefuted", '{"verdict":"Supported"}']:
            self.assertIsNone(verdict_from_text(answer))

    def test_plain_transport_does_not_claim_a_prefix_was_supplied(self):
        messages = [{"role": "system", "content": "Verify."},
                    {"role": "user", "content": "Inspect the image."}]
        plain = portable_tools(messages, [], assistant_prefix=False)
        prefixed = portable_tools(messages, [], assistant_prefix=True)
        self.assertNotIn("prefix below", plain[-1]["content"])
        self.assertIn("prefix below", prefixed[-1]["content"])
        self.assertEqual(messages[-1]["content"], "Inspect the image.")

    def test_tool_transport_does_not_repair_wrong_function_or_arguments(self):
        call, text = parse_tool_call(
            '<tool_call>{"name":"inspect","arguments":{"image_id":"actual"}}</tool_call>')
        self.assertEqual(call["arguments"]["image_id"], "actual")
        self.assertEqual(text, "")
        for raw in [
            '<tool_call>{"tool":"inspect","arguments":{"image_id":"actual"}}</tool_call>',
            '<tool_call>{"name":"inspect","arguments":null}</tool_call>',
            '<tool_call>{"name":"inspect","arguments":{}}',
        ]:
            with self.assertRaises(ValueError):
                parse_tool_call(raw)


if __name__ == "__main__":
    unittest.main()

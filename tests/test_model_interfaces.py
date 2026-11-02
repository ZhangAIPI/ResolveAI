"""Interface failures must remain distinguishable from model abstention."""
import unittest
from scripts.check_model_interfaces import verdict_from_text
from resolveai.vlm_worker import parse_tool_call, portable_tools, decode_tool_call


class InterfaceTests(unittest.TestCase):
    def test_qwen_xml_preserves_strings_structured_values_and_one_call_limit(self):
        tools = [{"function": {"name": "finish", "parameters": {"properties": {
            "verdict": {"type": "string"}, "citations": {"type": "array"},
        }}}}]
        raw = ('I can decide.<tool_call><function=finish>'
               '<parameter=verdict>Refuted</parameter>'
               '<parameter=citations>[{"image_id":"visible","bbox":[0,0,20,30]}]</parameter>'
               '</function></tool_call>')
        call, text = decode_tool_call(raw, tools, "native", "qwen3_5", True)
        self.assertEqual(text, "I can decide.")
        self.assertEqual(call, {"name": "finish", "arguments": {
            "verdict": "Refuted", "citations": [{"image_id": "visible", "bbox": [0, 0, 20, 30]}],
        }})
        for invalid in [
            raw + raw,
            raw.replace("</function>", "<parameter=verdict>Supported</parameter></function>"),
            raw.replace("[{", "not JSON [{"),
            raw.replace("</function>", "unparsed text</function>"),
            raw.replace("parameter=verdict", "parameter=unknown"),
            raw.replace("function=finish", "function=invented"),
            raw.replace("</tool_call>", ""),
        ]:
            with self.subTest(raw=invalid):
                with self.assertRaises(ValueError):
                    decode_tool_call(invalid, tools, "native", "qwen3_5", True)
        with self.assertRaises(ValueError):
            decode_tool_call(raw, tools, "native", "qwen3_vl", True)
        json_call, _ = decode_tool_call(
            '<tool_call>{"name":"finish","arguments":{"verdict":"Refuted"}}</tool_call>',
            tools, "native", "qwen3_5", True)
        self.assertEqual(json_call["arguments"], {"verdict": "Refuted"})

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

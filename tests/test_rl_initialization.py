"""GRPO must use the same transport as evaluation and keep RL-only initialization unadapted."""
from types import SimpleNamespace
import unittest
import torch
from scripts.train_online_grpo import policy_inputs, assert_unadapted


class InputsTests(unittest.TestCase):
    def processor(self, native):
        class Processor:
            chat_template = "tools tool_calls" if native else "plain chat"
            def apply_chat_template(self, messages, **kwargs):
                self.seen = messages, kwargs
                return {"input_ids":torch.tensor([[1,2]]),"attention_mask":torch.ones(1,2,dtype=torch.long),
                        "mm_token_type_ids":torch.ones(1,2,dtype=torch.long)}
            def tokenizer(self, text, **kwargs):
                self.prefix = text
                return SimpleNamespace(input_ids=torch.tensor([[8,9]]))
        return Processor()

    def test_portable_prefix_is_input_not_an_action_target(self):
        processor=self.processor(False)
        public={"messages":[{"role":"system","content":"Public instructions"},
                            {"role":"user","content":"Visible evidence"}],
                "tools":[{"type":"function","function":{"name":"finish","parameters":{}}}]}
        batch,prefix,adapter=policy_inputs(processor,public)
        self.assertEqual(adapter,"portable")
        self.assertEqual(prefix,'<tool_call>\n{"name":')
        self.assertEqual(batch["input_ids"].tolist(),[[1,2,8,9]])
        self.assertEqual(batch["mm_token_type_ids"].tolist(),[[1,1,0,0]])
        self.assertEqual(batch["attention_mask"].shape,batch["input_ids"].shape)
        self.assertEqual(public["messages"][0]["content"],"Public instructions")
        self.assertIsNone(processor.seen[1]["tools"])

    def test_native_prompt_and_tools_are_preserved(self):
        processor=self.processor(True)
        public={"messages":[{"role":"user","content":"Visible evidence"}],"tools":[]}
        batch,prefix,adapter=policy_inputs(processor,public)
        self.assertEqual((prefix,adapter),("","native"))
        self.assertEqual(batch["input_ids"].tolist(),[[1,2]])
        self.assertEqual(processor.seen[1]["tools"],[])

    def test_nonzero_lora_cannot_pass_as_a_pretrained_base(self):
        class Policy:
            def __init__(self,value): self.value=value
            def named_parameters(self):
                return [("language_model.lora_B.default.weight",torch.tensor([self.value]))]
        assert_unadapted(Policy(0.))
        with self.assertRaisesRegex(ValueError,"zero-output"):
            assert_unadapted(Policy(.5))

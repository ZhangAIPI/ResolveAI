"""Check policy-gradient direction, PPO clipping, and the hidden reward boundary."""
from copy import deepcopy
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from resolveai.conversation import Conversation
from resolveai.environment import Environment
from resolveai.policy_gradient import group_advantages, grpo_token_loss, trajectory_reward
from scripts.build_condition_sft import make_case


@unittest.skipUnless(find_spec("torch"), "model dependencies are optional")
class PolicyLossTests(unittest.TestCase):
    def test_group_centering_and_no_variance(self):
        import torch
        self.assertTrue(torch.equal(group_advantages([1.,1.,1.,1.]),torch.zeros(4)))
        values = group_advantages([-1.,1.,-1.,1.])
        self.assertAlmostEqual(values.mean().item(),0)
        self.assertAlmostEqual(values.std(unbiased=False).item(),1)

    def test_cost_differences_are_not_amplified_to_accuracy_differences(self):
        small = group_advantages([.94,1.0]).abs().max().item()
        large = group_advantages([-1.0,1.0]).abs().max().item()
        self.assertLess(small,large/20)

    def test_reward_advantage_moves_probability_in_the_expected_direction(self):
        import torch
        for advantage in [-1.,1.]:
            logits = torch.nn.Parameter(torch.zeros(2))
            optimizer = torch.optim.SGD([logits],lr=.1)
            logp = logits.log_softmax(0)[0:1]
            old = logp.detach().clone()
            loss,kl = grpo_token_loss(logp,old,old,advantage,beta=0)
            self.assertEqual(kl.item(),0)
            loss.mean().backward()
            optimizer.step()
            self.assertGreater((logits.softmax(0)[0].item()-.5)*advantage,0)

    def test_clipping_and_reference_are_not_optimized(self):
        import torch
        logp = torch.tensor([0.],requires_grad=True)
        old = torch.tensor([-1.],requires_grad=True)
        reference = torch.tensor([0.],requires_grad=True)
        loss,_ = grpo_token_loss(logp,old,reference,1.,beta=0)
        loss.sum().backward()
        self.assertEqual(logp.grad.item(),0)
        self.assertIsNone(old.grad)
        self.assertIsNone(reference.grad)
        with self.assertRaises(ValueError):
            grpo_token_loss(torch.zeros(2),torch.zeros(1),torch.zeros(2),1.)


class RewardBoundaryTests(unittest.TestCase):
    def test_truth_and_reward_stay_private_and_a_wrong_or_unsupported_finish_is_not_success(self):
        class NoOCR:
            available = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("target.png","reference.png"):
                Image.new("RGB",(32,32),"gray").save(root/name)
            case,correct,_ = make_case({"image":"target.png","label":"normal"},"reference.png",root)
            session = Conversation(Environment(case,root,ocr_backend=NoOCR()))
            public = json.dumps(session.public())
            for key in ('"annotation"', '"expected_verdict"', '"grounded_correct"', 'source-derived-unreviewed'):
                self.assertNotIn(key,public)
            session.call(correct)
            self.assertEqual(trajectory_reward(session),(1.,True))
            for mutation in ("wrong","unsupported"):
                session = Conversation(Environment(case,root,ocr_backend=NoOCR()))
                call = deepcopy(correct)
                if mutation=="wrong":
                    call["arguments"]["verdict"]="Supported"
                else:
                    call["arguments"]["citations"]=[]
                session.call(call)
                self.assertEqual(trajectory_reward(session),(-1.,False))
            missing = deepcopy(case)
            missing["initial"] = [case["evidence"][1]["id"]]
            missing["evidence"] = missing["evidence"][1:]
            session = Conversation(Environment(missing,root,ocr_backend=NoOCR()))
            session.call({"name":"finish","arguments":{"verdict":"Need more evidence","citations":[],"links":[]}})
            self.assertEqual(trajectory_reward(session),(1.,True))


if __name__ == "__main__":
    unittest.main()

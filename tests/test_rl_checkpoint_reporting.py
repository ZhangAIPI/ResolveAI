"""Final-checkpoint views preserve training identity and reject unfinished runs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.adaptation_shards import prepare_final_run


class FinalCheckpointTests(unittest.TestCase):
    def make_run(self, root, group=64, updates=2):
        run = root / "training"
        (run/"latest").mkdir(parents=True)
        state = {"group":group,"optimizer_updates":updates,"best_group":0}
        (run/"latest/state.json").write_text(json.dumps(state))
        (run/"latest/adapter_model.safetensors").write_bytes(b"final-weights")
        (run/"complete.json").write_text(json.dumps({"state":state,"adapter":"best","adapter_sha256":"old"}))
        (run/"training_manifest.json").write_text(json.dumps({"configuration":{"groups":64},"data":{}}))
        return run

    def test_final_view_reuses_weights_and_preserves_original_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            run=self.make_run(Path(temp))
            original=(run/"complete.json").read_bytes()
            output=prepare_final_run(run)
            complete=json.loads((output/"complete.json").read_text())
            self.assertEqual(complete["adapter"],str(run/"latest"))
            self.assertEqual(complete["adapter_sha256"],hashlib.sha256(b"final-weights").hexdigest())
            self.assertEqual(complete["evaluated_group"],64)
            self.assertEqual((run/"complete.json").read_bytes(),original)
            self.assertFalse((output/"latest").exists())
            self.assertEqual(prepare_final_run(run),output)

    def test_unfinished_or_unupdated_training_is_not_a_completed_rl_result(self):
        for group,updates in ((63,2),(64,0)):
            with self.subTest(group=group,updates=updates), tempfile.TemporaryDirectory() as temp:
                run=self.make_run(Path(temp),group,updates)
                with self.assertRaisesRegex(ValueError,"completed training"):
                    prepare_final_run(run)

    def test_changed_final_weights_cannot_silently_replace_an_evaluated_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            run=self.make_run(Path(temp))
            prepare_final_run(run)
            (run/"latest/adapter_model.safetensors").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError,"identity differs"):
                prepare_final_run(run)

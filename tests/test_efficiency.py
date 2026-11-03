"""Cost accounting must retain failures and count only actual model work."""
import unittest
from scripts.score_efficiency import episode_usage, summarize


class EfficiencyTests(unittest.TestCase):
    def test_context_rejection_is_not_model_prefill_and_failures_stay_in_mean(self):
        failed = {"task":"state","verdict":None,"source_label":"Supported",
                  "termination":"context_limit","latency_s":5.0,"tool_calls":1,"requests":0}
        trace = {"wall_latency_s":5.0,"generation":[
            {"input_tokens":100,"output_tokens":10,"latency_s":1.0},
            {"input_tokens":200,"output_tokens":20,"latency_s":2.0,"parse_error":True},
            {"halt":"context_limit","input_tokens":9999}]}
        first = episode_usage(failed, trace)
        abstained = {**failed,"verdict":"Need more evidence","termination":"finished",
                     "latency_s":1.0,"tool_calls":0}
        second = episode_usage(abstained, {"wall_latency_s":1.0,"generation":[
            {"input_tokens":100,"output_tokens":4,"latency_s":.5}]})
        result = summarize([first,second])
        self.assertEqual(result["episodes"],2)
        self.assertEqual(result["accuracy_pct"],0)
        self.assertEqual(result["finished_pct"],50)
        self.assertEqual(result["mean_wall_s"],3)
        self.assertEqual(result["mean_input_tokens"],200)
        self.assertEqual(result["mean_output_tokens"],17)
        self.assertEqual(result["mean_model_calls"],1.5)
        self.assertEqual(result["mean_rejected_calls"],.5)

    def test_missing_measurement_is_not_zero(self):
        row = {"task":"identity","verdict":"Supported","source_label":"Supported",
               "termination":"finished","latency_s":1.0,"tool_calls":0,"requests":0}
        with self.assertRaises(KeyError):
            episode_usage(row, {"wall_latency_s":1.0,"generation":[
                {"input_tokens":20,"latency_s":.5}]})
        with self.assertRaisesRegex(ValueError,"latency"):
            episode_usage(row, {"wall_latency_s":2.0,"generation":[]})

    def test_rejected_prompt_cannot_have_generated_tokens(self):
        with self.assertRaisesRegex(ValueError,"rejected"):
            episode_usage({}, {"generation":[
                {"halt":"context_limit","input_tokens":10000,"output_tokens":1}]})


if __name__ == "__main__":
    unittest.main()

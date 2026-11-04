"""Only a complete matched cohort may add a training row to the paper."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from scripts.watch_adaptation_results import complete_cohort, replace_block
from scripts.score_sft_comparison import render_adaptations, initial_rl_checkpoint


class PublicationTests(unittest.TestCase):
    def test_an_incomplete_access_version_blocks_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"manifest.json").write_text("{}")
            rows = [{"family_id":"a","variant":v,"arm":"interactive_agent"} for v in
                    ["Sufficient","Obtainable","Missing","Unavailable"]]
            rows += [{"family_id":"a","variant":"Sufficient","arm":a} for a in ["initial_text","full_available_text"]]
            with sqlite3.connect(root/"episodes.sqlite") as db:
                db.execute("CREATE TABLE episodes(result)")
                db.executemany("INSERT INTO episodes VALUES(?)",[(json.dumps(r),) for r in rows[:-1]])
            self.assertFalse(complete_cohort(root,["a"]))
            with sqlite3.connect(root/"episodes.sqlite") as db:
                db.execute("INSERT INTO episodes VALUES(?)",(json.dumps(rows[-1]),))
            self.assertTrue(complete_cohort(root,["a"]))

    def test_final_checkpoint_is_distinct_from_a_validation_retained_initialization(self):
        selected = {"training":{"configuration":{"groups":64}},
                    "checkpoint":{"state":{"best_group":0}}}
        self.assertTrue(initial_rl_checkpoint(selected))
        final = {**selected, "checkpoint":{**selected["checkpoint"],"evaluated_group":64}}
        self.assertFalse(initial_rl_checkpoint(final))
        self.assertEqual(selected["checkpoint"]["state"]["best_group"],0)


    def test_result_block_preserves_surrounding_edits(self):
        text = "Author introduction\n\\section*{LLM Usage Statement}\nAuthor end"
        first = replace_block(text,"First measurements")
        second = replace_block(first,"Updated measurements")
        self.assertIn("Author introduction",second)
        self.assertIn("Author end",second)
        self.assertNotIn("First measurements",second)
        self.assertEqual(second.count("% adaptation-results-start"),1)

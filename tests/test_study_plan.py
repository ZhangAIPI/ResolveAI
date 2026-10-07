"""Assignments protect blinding and prevent remembering source objects."""

from collections import Counter
import unittest
from resolveai.study_plan import assign, select_families, short_assign


class StudyTests(unittest.TestCase):
    def test_source_disjoint_and_arm_balance(self):
        families = [
            {
                "family_id": str(i),
                "task": "state" if i < 6 else "identity",
                "provenance": {},
                "group_ids": ["source-" + str(i)],
            }
            for i in range(12)
        ]
        queues, variants = assign(families, participants=6, repeats=2, seed=7)
        seen = Counter()
        for queue in queues:
            self.assertEqual(len(queue), 12)
            self.assertEqual(len({r["family_id"] for r in queue}), 12)
            self.assertEqual(
                Counter(r["condition"] for r in queue),
                {"initial": 4, "interactive": 4, "full_available": 4},
            )
            for row in queue:
                seen[(row["family_id"], row["condition"])] += 1
        self.assertTrue(all(n == 2 for n in seen.values()))

    def test_twenty_people_get_three_conditions_and_independent_micro_reviews(self):
        families = []
        for index in range(5):
            for label in ("Supported", "Refuted"):
                families.append(
                    {
                        "family_id": f"state-{index}-{label}",
                        "task": "state",
                        "category": f"category-{index}",
                        "provenance": {"criterion": f"condition-{index}"},
                        "provenance_truth": label,
                        "group_ids": [f"state-{index}-{label}"],
                    }
                )
        for index in range(4):
            for kind in ("source-positive", "easy-negative"):
                families.append(
                    {
                        "family_id": f"identity-{index}-{kind}",
                        "task": "identity",
                        "category": f"identity-category-{index}",
                        "provenance": {"construction": kind},
                        "evidence": [{"source_id": f"anchor-{index}"}],
                        "group_ids": [f"anchor-{index}"]
                        + ([f"foreign-{index}"] if kind == "easy-negative" else []),
                    }
                )
        for index in range(2):
            families.append(
                {
                    "family_id": f"unknown-{index}",
                    "task": "identity",
                    "category": f"unknown-category-{index}",
                    "provenance": {"construction": "same-category-candidate"},
                    "group_ids": [f"unknown-{index}"],
                }
            )
        queues, variants = short_assign(families)
        by_family = {c["family_id"]: c for c in families}
        reviews = Counter()
        searches = Counter()
        self.assertEqual(len(queues), 20)
        for queue in queues:
            self.assertEqual(len(queue), 5)
            self.assertEqual(
                len({by_family[r["family_id"]]["category"] for r in queue}), 5
            )
            self.assertEqual(
                Counter(r["condition"] for r in queue if r["mode"] == "search"),
                {"initial": 1, "interactive": 1, "full_available": 1},
            )
            used = set()
            for row in queue:
                groups = set(by_family[row["family_id"]]["group_ids"])
                self.assertFalse(used & groups)
                used |= groups
                if row["mode"] == "review":
                    reviews[row["family_id"]] += 1
                else:
                    searches[(row["family_id"], row["condition"])] += 1
        self.assertEqual(len(reviews), 20)
        self.assertTrue(all(n == 2 for n in reviews.values()))
        self.assertEqual(len(searches), 30)
        self.assertTrue(all(n == 2 for n in searches.values()))
        # A live study may already have responses: only pending participants move,
        # and the frozen family/arm counts and existing task order must survive.
        migrated, migrated_variants = short_assign(
            families,
            locked={0: queues[0]},
            searching_families={
                r["family_id"] for q in queues for r in q if r["mode"] == "search"
            },
        )
        self.assertEqual(migrated[0], queues[0])
        self.assertEqual(migrated_variants, variants)
        self.assertEqual(
            Counter(
                (r["family_id"], r.get("condition"), r["mode"])
                for q in migrated
                for r in q
            ),
            Counter(
                (r["family_id"], r.get("condition"), r["mode"])
                for q in queues
                for r in q
            ),
        )
        for queue in migrated[1:]:
            self.assertEqual(
                len({by_family[r["family_id"]]["category"] for r in queue}), 5
            )

    def test_infeasible_shared_sources_are_rejected(self):
        families = [
            {
                "family_id": str(i),
                "task": "state",
                "provenance": {},
                "group_ids": ["same"],
            }
            for i in range(6)
        ]
        with self.assertRaisesRegex(ValueError, "source"):
            assign(families, participants=6, repeats=2)


if __name__ == "__main__":
    unittest.main()

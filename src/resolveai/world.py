"""Immutable hidden scene plus rendered/captured evidence pool.

The world model is state-driven, not generative image editing. Render complete
camera/time libraries offline; agent requests only release matching pixels.
Identity, condition and annotation never appear in tool observations.
"""
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class EvidenceWorld:
    scene_json: str
    evidence_json: str

    @classmethod
    def from_case(cls, case):
        return cls(json.dumps(case.get("scene", {}), sort_keys=True),
                   json.dumps(case["evidence"], sort_keys=True))

    @property
    def fingerprint(self):
        """Evaluator-only invariant for branch audits, never a model observation."""
        return hashlib.sha256((self.scene_json + self.evidence_json).encode()).hexdigest()

    def select(self, query, released):
        matches = [row for row in json.loads(self.evidence_json)
                   if row["available"] and all(row[k] == v for k, v in query.items())]
        matches.sort(key=lambda row: row["id"])
        return next((r for r in matches if r["id"] not in released), matches[0] if matches else None)

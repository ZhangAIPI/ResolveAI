"""Audited continuation after a client fix; completed answers stay immutable."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from resolveai.benchmark import locked_manifest

FINANCIAL_FIELDS = {
    "ledger", "ledger_cap_usd", "stage_cap_usd", "budget_stage",
    "source_sha256", "script_sha256", "continuation",
}


def row_digests(database):
    with sqlite3.connect(Path(database).resolve().as_uri()+"?mode=ro", uri=True) as db:
        return sorted(hashlib.sha256((row+"\0"+trace).encode()).hexdigest()
                      for row, trace in db.execute("SELECT result,trace FROM episodes"))


def continue_manifest(path, configuration):
    """Allow finance/client changes only; keep the first manifest and answer hashes."""
    path = Path(path)
    archive = path.with_name("manifest.before-client-fix.json")
    audit = path.with_name("answers.before-client-fix.json")
    previous = json.loads((archive if archive.exists() else path).read_text())
    old = previous["configuration"]
    stable = lambda config: {k:v for k,v in config.items() if k not in FINANCIAL_FIELDS}
    if stable(old) != stable(configuration):
        raise ValueError("Continuation changed model, task, tools, pixels or evaluation settings")
    current = row_digests(path.parent/"episodes.sqlite")
    if audit.exists():
        recorded = json.loads(audit.read_text())
        if not set(recorded) <= set(current):
            raise ValueError("A previously completed answer changed")
    else:
        recorded = current
        audit.write_text(json.dumps(recorded, indent=2)+"\n")
    if not archive.exists():
        shutil.copy2(path, archive)
    configuration = {**configuration, "continuation": {
        "reason": "Client input-count guard repair; completed answers are not rerun",
        "previous_manifest_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "previous_source_commit": previous["source_commit"],
        "preserved_answers": len(recorded),
        "preserved_answer_hashes_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    }}
    if json.loads(path.read_text())["configuration"] == configuration:
        return locked_manifest(path, configuration)
    pending = path.with_name("manifest.client-fix.pending.json")
    if pending.exists():
        pending.unlink()
    manifest = locked_manifest(pending, configuration)
    pending.replace(path)
    return manifest

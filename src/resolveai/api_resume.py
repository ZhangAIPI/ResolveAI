"""Audited client or finance continuation; completed answers stay immutable."""
import ast
from contextlib import closing
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
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+"?mode=ro", uri=True)) as db:
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


def answer_snapshot(directory):
    """Hash completed records and referenced images, without exposing responses."""
    directory = Path(directory)
    rows = row_digests(directory/"episodes.sqlite")
    images = {}
    with closing(sqlite3.connect((directory/"episodes.sqlite").resolve().as_uri()+"?mode=ro", uri=True)) as db:
        for raw, in db.execute("SELECT trace FROM episodes"):
            for message in json.loads(raw).get("messages", []):
                content = message.get("content", [])
                for block in content if isinstance(content, list) else []:
                    if block.get("type") != "image" or not block.get("image_ref"):
                        continue
                    relative = Path(block["image_ref"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Expected an immutable relative image reference")
                    images[str(relative)] = hashlib.sha256((directory/relative).read_bytes()).hexdigest()
    return {"rows": rows, "images": dict(sorted(images.items()))}


def verify_snapshot(directory, recorded):
    current = answer_snapshot(directory)
    if not set(recorded["rows"]) <= set(current["rows"]):
        raise ValueError("A previously completed answer changed")
    if any(current["images"].get(key) != value for key, value in recorded["images"].items()):
        raise ValueError("A previously completed answer image changed")
    return current


def inference_code_digest(repository):
    """Cover inference modules and evaluator execution, excluding finance helpers."""
    repository = Path(repository)
    source = repository/"src/resolveai"
    digest = hashlib.sha256()
    for path in sorted(source.glob("*.py")):
        if path.name in {"api_budget.py", "api_resume.py"}:
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((repository/"scripts/evaluate_study.py").read_bytes())
    tree = ast.parse((repository/"scripts/evaluate_openai.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("select_cases", "configure_api_key"):
        digest.update(ast.dump(functions[name], include_attributes=False).encode())
    body = functions["main"].body
    start = next(i for i, node in enumerate(body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ledger" for t in node.targets))
    for node in body[start:]:
        digest.update(ast.dump(node, include_attributes=False).encode())
    return digest.hexdigest()


def continue_budget_manifest(path, configuration, previous_repository, repository):
    """Permit only financing changes after checking the frozen inference code."""
    path = Path(path)
    previous_repository, repository = Path(previous_repository), Path(repository)
    tag = Path(configuration["ledger"]).stem
    archive = path.with_name("manifest.before-budget-"+tag+".json")
    audit = path.with_name("answers.before-budget-"+tag+".json")
    previous = json.loads((archive if archive.exists() else path).read_text())
    old = previous["configuration"]
    stable = lambda config: {key:value for key,value in config.items() if key not in FINANCIAL_FIELDS}
    if stable(old) != stable(configuration):
        raise ValueError("Budget continuation changed model, task, tools, pixels or evaluation settings")
    previous_source = hashlib.sha256()
    for source in sorted((previous_repository/"src/resolveai").glob("*.py")):
        previous_source.update(source.name.encode())
        previous_source.update(source.read_bytes())
    if previous_source.hexdigest() != old["source_sha256"]:
        raise ValueError("Frozen repository does not match the preceding source manifest")
    if hashlib.sha256((previous_repository/"scripts/evaluate_openai.py").read_bytes()).hexdigest() != old["script_sha256"]:
        raise ValueError("Frozen evaluator does not match the preceding manifest")
    inference_digest = inference_code_digest(previous_repository)
    if inference_digest != inference_code_digest(repository):
        raise ValueError("Budget continuation changed inference code")
    current = answer_snapshot(path.parent)
    if audit.exists():
        recorded = json.loads(audit.read_text())
        verify_snapshot(path.parent, recorded)
    else:
        recorded = current
        audit.write_text(json.dumps(recorded, indent=2)+"\n")
    if not archive.exists():
        shutil.copy2(path, archive)
    configuration = {**configuration, "continuation": {
        "reason": "Budget reallocation only; completed answers are not rerun",
        "previous_manifest_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "previous_source_commit": previous["source_commit"],
        "previous_repository": str(previous_repository),
        "inference_code_sha256": inference_digest,
        "preserved_answers": len(recorded["rows"]),
        "preserved_answer_hashes_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    }}
    if json.loads(path.read_text())["configuration"] == configuration:
        return locked_manifest(path, configuration)
    pending = path.with_name("manifest.budget-resume.pending.json")
    if pending.exists():
        pending.unlink()
    manifest = locked_manifest(pending, configuration)
    pending.replace(path)
    return manifest

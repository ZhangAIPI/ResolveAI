"""Publish measured SFT Table 2 after evaluation; refuse stale or edited paper sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, **kwargs):
    result = subprocess.run(args, check=True, capture_output=True, text=True, **kwargs)
    if result.stdout.strip():
        print(result.stdout[-1500:], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("repository", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--table1-sha256", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    args = parser.parse_args()
    if sha(Path(__file__)) != args.expected_script_sha256:
        raise ValueError("Publisher code changed after queuing")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Publish from the allocated GPU node")
    marker = args.run / "paper/published.json"
    if marker.exists():
        print(marker.read_text(), flush=True)
        return
    repo, runtime = args.repository, args.runtime
    report = json.loads((args.run / "paper/sft_results.json").read_text())
    cohort_path = repo / "docs/results/cvpr_20261102_candidate_scores.json"
    cohort = json.loads(cohort_path.read_text())
    from scripts.score_sft_comparison import compare, render
    updated = compare(cohort_path, runtime / "evaluation/cvpr-20261101/qwen3",
                      args.run / "benchmark", args.run)
    if updated["family_ids"] != report["family_ids"]:
        raise ValueError("Completed run and refreshed task breakdown differ")
    report = updated
    if report["family_ids"] != cohort["family_ids"] or report["dataset_sha256"] != cohort["dataset_sha256"]:
        raise ValueError("Paper cohort changed; retain measured artifacts for reconciliation")
    if sha(repo / "paper/tables/benchmark_results.tex") != args.table1_sha256:
        raise ValueError("Paper Table 1 changed")
    expected_table = render(report)
    table_path = repo / "paper/tables/sft_results.tex"
    status = command(["git", "status", "--porcelain"], cwd=repo).stdout.strip()
    recovering_table = status == "?? paper/tables/sft_results.tex" and (
        table_path.read_text() == expected_table)
    if status and not recovering_table:
        raise ValueError("Repository has edits; leave measured artifacts for review")
    mirror = runtime / "paper-overleaf"
    for path in (repo / "paper").rglob("*"):
        if path.is_file() and path.suffix in {".tex", ".bib", ".sty", ".bst", ".jpg", ".png", ".json", ".md"}:
            target = mirror / path.relative_to(repo / "paper")
            if recovering_table and path == table_path:
                continue
            if not target.exists() or sha(target) != sha(path):
                raise ValueError("Local paper and Overleaf mirror differ")
    cli = json.loads((runtime / "tools/olcli/invocation.json").read_text())["command"]
    project = "6ae80679a4c7a64df04ac5b8"
    command(cli + ["diff", project, str(mirror), "--name-only", "--exit-code"])
    table = expected_table
    with tempfile.TemporaryDirectory(prefix="resolveai-sft-paper-") as directory:
        stage = Path(directory) / "paper"
        shutil.copytree(repo / "paper", stage)
        (stage / "tables/sft_results.tex").write_text(table)
        command([str(runtime / "tools/tex/tectonic"), "--outdir", directory, "main.tex"],
                cwd=stage, env={**os.environ, "XDG_CACHE_HOME": str(runtime / "tools/tex/cache")})
        sys.path.insert(0, str(runtime / "tools/pdf"))
        import fitz
        document = fitz.open(Path(directory) / "main.pdf")
        text = "\n".join(page.get_text() for page in document)
        if "[?]" in text or "\ufffd" in text or "Table 2." not in text or document.metadata.get("author"):
            raise ValueError("Compiled paper failed reference/table/anonymity checks")
        (repo / "paper/tables/sft_results.tex").write_text(table)
    result_path = repo / "docs/results/cvpr_sft_candidate_scores.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    command(["git", "diff", "--check"], cwd=repo)
    command(["git", "add", "paper/tables/sft_results.tex",
             "docs/results/cvpr_sft_candidate_scores.json"], cwd=repo)
    command(["git", "commit", "-m", "Add measured source-disjoint SFT comparison as paper Table 2"], cwd=repo)
    command(["git", "push", "origin", "main"], cwd=repo)
    destination = mirror / "tables/sft_results.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / "paper/tables/sft_results.tex", destination)
    command(cli + ["push", str(mirror)])
    command(cli + ["pdf", project, "-o", str(runtime / "paper-draft/ResolveAI_CVPR_methods.pdf"),
                   "-r", "main.tex"])
    command(cli + ["diff", project, str(mirror), "--name-only", "--exit-code"])
    shutil.copy2(runtime / "paper-draft/ResolveAI_CVPR_methods.pdf",
                 "/u/zzh136/Documents/Codex/ResolveAI_CVPR_methods.pdf")
    marker.write_text(json.dumps({
        "commit": command(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip(),
        "table2_sha256": sha(destination), "families": report["common_families"],
        "overleaf": project, "status": "published measured Table 2",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

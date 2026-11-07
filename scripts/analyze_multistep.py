"""Complete-only, bank-clustered analysis of the procedural two-request suite."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np

from resolveai.grounding import link_matches

POLICIES = ("adaptive_acquisition", "fixed_restoration", "fixed_view_order",
            "random_request", "oracle")
SCORE_KEYS = ("procedural_evidence_sufficient", "chain_grounded_correct",
              "localized_grounded_correct", "source_label_correct")
METRICS = (*SCORE_KEYS, "finished")
SEED = 20261107
REPLICATES = 20000
CHOICE_EXPERIMENT = "procedural-multistep-interface-assisted-choice-v1"
CHOICE_SCOPE = "interface-assisted-finite-choice-diagnostic"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path):
    source = Path(path)
    source = source / "cases.json" if source.is_dir() else source
    document = json.loads(source.read_text())
    rows = document["cases"] if isinstance(document, dict) else document
    manifest_path = source.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset_sha256"] != digest(source) or manifest["annotation_status"] != "procedural":
        raise ValueError("Dataset manifest checksum or procedural scope mismatch")
    cases = [c for c in rows if c.get("variant") == "Obtainable"]
    indexed = {c["case_id"]: c for c in cases}
    if not cases or len(indexed) != len(cases):
        raise ValueError("The Obtainable cohort is empty or duplicated")
    groups = defaultdict(list)
    for case in cases:
        if case["task"] != "identity_condition" or not case.get("paired_history_id"):
            raise ValueError("Only the standalone procedural identity-condition track is supported")
        groups[case["paired_history_id"]].append(case)
    for pair in groups.values():
        if (len(pair) != 2 or len({c["family_id"] for c in pair}) != 2
                or len({c["multistep_certificate"]["bank_sha256"] for c in pair}) != 1
                or len({c["multistep_certificate"]["target_rear_view"] for c in pair}) != 2
                or {c["provenance_truth"] for c in pair} != {"Supported", "Refuted"}):
            raise ValueError("A scene bank must contain two distinct matched positions with opposite labels")
    if manifest.get("paired_scene_banks") != len(groups):
        raise ValueError("The cohort does not match the declared number of paired banks")
    return source, indexed, groups, manifest



def _track(config):
    experiment = config.get("experiment", "")
    if experiment == CHOICE_EXPERIMENT:
        if (config.get("interface_scope") != CHOICE_SCOPE
                or config.get("native_grounding_claim") is not False):
            raise ValueError("Assisted-choice scope must be explicit and cannot claim native grounding")
        return "assisted_finite_choice"
    if (experiment.startswith("procedural-multistep-two-request-")
            and "assisted" not in config.get("interface_scope", "")
            and not config.get("automatic_identity_links", False)):
        return "native"
    raise ValueError("Unsupported or inconsistent experiment/interface scope")


def _identity_links_valid(case, decision):
    if not isinstance(decision, dict) or decision.get("verdict") not in {"Supported", "Refuted"}:
        return False
    expected = case["multistep_certificate"].get("required_links", [])
    links = decision.get("links", [])
    try:
        return bool(expected) and all(any(link_matches(e, a) for a in links) for e in expected) and all(
            any(link_matches(e, a) for e in expected) for a in links)
    except (KeyError, TypeError, ValueError):
        return False

def _common_config(config):
    result = {k: v for k, v in config.items() if k not in {
        "dataset", "model", "model_id", "revision", "families", "case_ids", "shards", "shard"}}
    if "assets" in result:
        result["assets"] = sorted((v["sha256"], v["bytes"]) for v in result["assets"].values())
    return result


def _manifest_paths(root):
    root = Path(root)
    if (root / "manifest.json").is_file():
        return [root / "manifest.json"]
    return sorted(root.glob("shard-*/manifest.json"))


def _read_rows(path):
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='episodes'").fetchone():
            return []
        return list(db.execute("SELECT case_id,policy,result,trace FROM episodes"))


def _requests(case, trace):
    allowed = case["request_options"]
    executed = trace["acquisition"]["executed_requests"]
    requests = []
    for entry in executed:
        call = entry["call"]
        query = call["arguments"]["query"]
        if (call["name"] != "request_photo" or set(query) != {"object", "time", "view"}
                or any(query[k] not in allowed[k + "s"] for k in ("object", "time", "view"))):
            raise ValueError("A recorded request is outside the public options")
        if not entry.get("error"):
            if entry.get("status") not in {"provided", "unable_to_provide"}:
                raise ValueError("Executed request has no recorded environment status")
            requests.append(query["view"])
    return requests


def read_run(root, cases, source_hash):
    records, configurations, manifests, initial_hashes = {}, [], [], {}
    all_families = sorted({c["family_id"] for c in cases.values()})
    for path in _manifest_paths(root):
        manifest = json.loads(path.read_text())
        config = manifest["configuration"]
        track = _track(config)
        required = {"dataset_sha256", "model", "source_sha256", "script_sha256",
                    "budget", "max_new_tokens", "max_context_tokens", "seed",
                    "request_slots", "families", "case_ids", "policies", "variants",
                    "shard", "shards"}
        if not required <= config.keys():
            raise ValueError("Run manifest lacks frozen cohort/model/inference settings")
        if (config["dataset_sha256"] != source_hash
                or config.get("variants") != ["Obtainable"]
                or tuple(config["policies"]) != POLICIES
                or config.get("request_slots") != 2):
            raise ValueError("Run dataset, policies, variant or experiment mismatch")
        shard, shards = config["shard"], config["shards"]
        if type(shards) is not int or type(shard) is not int or not 0 <= shard < shards:
            raise ValueError("Invalid shard identity")
        families = all_families[shard::shards]
        ids = sorted(k for k, c in cases.items() if c["family_id"] in families)
        if sorted(config["families"]) != families or sorted(config["case_ids"]) != ids:
            raise ValueError("Run shard differs from the frozen cohort")
        if configurations:
            old = configurations[0]
            if (_common_config(old) != _common_config(config)
                    or old["model"] != config["model"] or old.get("model_id") != config.get("model_id")
                    or old.get("revision") != config.get("revision") or old["shards"] != shards):
                raise ValueError("Shards mix weights, data, source or inference settings")
        if any(c["shard"] == shard for c in configurations):
            raise ValueError("Duplicate shard manifest")
        configurations.append(config)
        manifests.append({"path": str(path), "sha256": digest(path),
                          "source_commit": manifest.get("source_commit")})
        for case_id, policy, raw, trace_raw in _read_rows(path.parent / "episodes.sqlite"):
            key = case_id, policy
            if case_id not in ids or policy not in POLICIES or key in records:
                raise ValueError("Unexpected or duplicate episode key")
            row, trace = json.loads(raw), json.loads(trace_raw)
            if ((row["case_id"], row["policy"]) != key
                    or row["family_id"] != cases[case_id]["family_id"]
                    or sha256(canonical(trace).encode()).hexdigest() != row["trace_sha256"]
                    or row["multistep_evaluation"] != trace["multistep_evaluation"]):
                raise ValueError("Result/trace checksum, key or scoring mismatch")
            row["_record_sha256"] = sha256(canonical(row).encode()).hexdigest()
            score = row["multistep_evaluation"]
            for name in (*SCORE_KEYS, "simulator_grounded_correct", "unsupported_decision"):
                if type(score.get(name)) is not bool:
                    raise ValueError("Missing or nonboolean frozen evaluation field")
            if (score["localized_grounded_correct"] != score["simulator_grounded_correct"]
                    or score["localized_grounded_correct"] > score["chain_grounded_correct"]
                    or score["chain_grounded_correct"] > score["source_label_correct"]
                    or (score["source_label_correct"] and not row["finished"])):
                raise ValueError("Inconsistent procedural metric nesting or completion")
            if track == "assisted_finite_choice":
                if row.get("interface_scope") != CHOICE_SCOPE:
                    raise ValueError("Assisted row has the wrong interface scope")
                for name in ("choice_valid", "front_slot_assessable", "selected_chain_target_correct"):
                    if type(row.get(name)) is not bool:
                        raise ValueError("Missing assisted-choice evaluation fields")
                if row["selected_chain_target_correct"] and not row["front_slot_assessable"]:
                    raise ValueError("An unassessable front slot cannot count as a correct match")
            row["_identity_links_valid"] = _identity_links_valid(cases[case_id], trace.get("decision"))
            row["_definite_finished"] = bool(row["finished"] and trace.get("decision")
                and trace["decision"].get("verdict") in {"Supported", "Refuted"})
            views = _requests(cases[case_id], trace)
            if len(views) != row["requests"] or len(views) > 2:
                raise ValueError("Recorded request count disagrees with real environment calls")
            row["_views"] = views
            row["_target"] = cases[case_id]["multistep_certificate"]["target_rear_view"]
            row["_duplicate_requests"] = len(views) - len(set(views))
            row["_front_then_target"] = (
                row["selection_complete"] and views == ["front_survey", row["_target"]])
            old = initial_hashes.setdefault(case_id, row["initial_public_sha256"])
            if old != row["initial_public_sha256"]:
                raise ValueError("Policies received different initial public observations")
            records[key] = row
    expected = {(case_id, policy) for case_id in cases for policy in POLICIES}
    if not set(records) <= expected:
        raise ValueError("Unplanned result keys")
    return {"records": records, "config": configurations[0] if configurations else None,
            "manifests": manifests, "expected": len(expected), "complete": set(records) == expected,
            "track": _track(configurations[0]) if configurations else None,
            "ordered_episode_sha256": sha256(canonical([
                [*key, records[key]["_record_sha256"]] for key in sorted(records)]).encode()).hexdigest()}


def bank_bootstrap(differences, case_groups, *, seed=SEED, replicates=REPLICATES):
    """Resample whole paired banks, retaining all cases in each sampled bank."""
    if set(differences) != set(case_groups) or not differences or replicates < 1:
        raise ValueError("Paired differences must cover exactly the frozen cohort")
    sums, sizes = defaultdict(float), Counter()
    for case_id, value in differences.items():
        group = case_groups[case_id]
        sums[group] += value
        sizes[group] += 1
    names = sorted(sums)
    totals = np.array([sums[k] for k in names])
    counts = np.array([sizes[k] for k in names])
    draws = np.random.default_rng(seed).integers(0, len(names), size=(replicates, len(names)))
    sampled = totals[draws].sum(axis=1) / counts[draws].sum(axis=1) * 100
    return {"difference_pp": float(sum(differences.values()) / len(differences) * 100),
            "paired_bank_bootstrap_ci95_pp": [float(v) for v in np.quantile(sampled, [.025, .975])],
            "banks": len(names), "episodes": len(differences)}


def _metric(row, key):
    if key in ("finished", "choice_valid", "selected_chain_target_correct"):
        return bool(row[key])
    return row["multistep_evaluation"][key]


def _paired_behavior(records, cases, groups, policy):
    counts = Counter()
    for pair in groups.values():
        rows = [records[(c["case_id"], policy)] for c in pair]
        if len({r["initial_public_sha256"] for r in rows}) != 1:
            raise ValueError("Matched histories did not share their initial public input")
        a, b = [r["_views"] for r in rows]
        counts["banks"] += 1
        counts["same_first_request"] += bool(a and b and a[0] == b[0])
        both_front = bool(a and b and a[0] == b[0] == "front_survey")
        counts["both_first_front"] += both_front
        different = bool(len(a) == len(b) == 2 and a[1] != b[1])
        counts["second_views_differ"] += different
        correct = bool(both_front and different and all(r["_front_then_target"] for r in rows))
        counts["both_second_requests_match_observed_targets"] += correct
    n = counts["banks"]
    front = counts["both_first_front"]
    return {"counts": dict(counts),
            "correct_switch_pct_all_banks": 100 * counts["both_second_requests_match_observed_targets"] / n,
            "correct_switch_pct_given_both_front": (
                100 * counts["both_second_requests_match_observed_targets"] / front if front else None),
            "conditional_denominator": "Banks where both matched histories first request front_survey",
            "initial_observations_identical": True}


def analyze(data, runs, *, seed=SEED, replicates=REPLICATES):
    source, cases, groups, dataset_manifest = load_dataset(data)
    if not runs:
        raise ValueError("Specify the complete planned set of model runs")
    loaded = {name: read_run(root, cases, digest(source)) for name, root in runs.items()}
    configs = [r["config"] for r in loaded.values() if r["config"] is not None]
    tracks = {r["track"] for r in loaded.values() if r["track"] is not None}
    if len(tracks) > 1:
        raise ValueError("Native and assisted-choice interfaces must be analyzed in separate reports")
    track = next(iter(tracks), None)
    metrics = (("procedural_evidence_sufficient", "source_label_correct", "choice_valid",
                "selected_chain_target_correct", "finished") if track == "assisted_finite_choice" else METRICS)
    if any(_common_config(c) != _common_config(configs[0]) for c in configs[1:]):
        raise ValueError("Models differ in cohort, source or inference settings")
    complete = all(r["complete"] for r in loaded.values())
    report = {
        "complete": complete, "scope": "Standalone procedural identity-condition search; Obtainable only",
        "independent_human_review": False, "interface_track": track,
        "native_grounding_claim": track == "native",
        "analysis_script_sha256": digest(__file__), "numpy_version": np.__version__,
        "settings": _common_config(configs[0]) if configs else None,
        "cohort": {"dataset_sha256": digest(source), "manifest_sha256": digest(source.parent / "dataset_manifest.json"),
                   "episodes_per_policy": len(cases), "paired_banks": len(groups),
                   "case_ids": sorted(cases), "renderer_source_sha256": dataset_manifest.get("renderer_source_sha256")},
        "bootstrap": {"unit": "paired_history_id / scene bank", "replicates": replicates,
                      "seed": seed, "method": "Paired percentile cluster bootstrap, central 95%; no multiplicity adjustment"},
        "denominator": "Every planned attempted episode, including invalid selections, failed submissions and abstentions",
        "definitions": {
            "chain_grounded_correct": "Correct source label, finite-world evidence identification and localized identity chain with diagnostic visible-mask coverage",
            "localized_grounded_correct": "Chain-correct with the additional diagnostic rectangle IoU/precision thresholds",
            "evidence_found": "Actual observed view hashes identify one source label in the finite world bank",
            "duplicate_requests": "Executed requests beyond the first request for each public view in the same episode",
            "efficiency": "All-attempt means; adaptive selector model calls count, scripted selectors generate no model tokens",
            "tokens": "Offered native input tokens and all generated tokens; unavailable if any transport accounting is incomplete",
        },
        "partial_scores_withheld": not complete, "cross_model_comparison_ready": complete,
        "completion_rule": "Each model reports scores only after its entire five-policy cohort is complete; cross-model comparison waits for all requested models",
        "models": {},
    }
    for name, run in loaded.items():
        records = run["records"]
        entry = {
            "complete": run["complete"], "completed_episodes": len(records), "expected_episodes": run["expected"],
            "ordered_episode_sha256": run["ordered_episode_sha256"],
            "completion_by_policy": {p: sum(k[1] == p for k in records) for p in POLICIES},
            "manifests": run["manifests"], "model_configuration": run["config"]["model"] if run["config"] else None,
        }
        if run["complete"]:
            entry["policies"] = {}
            for policy in POLICIES:
                rows = [records[(case_id, policy)] for case_id in sorted(cases)]
                n = len(rows)
                counts = {key: sum(_metric(r, key) for r in rows) for key in metrics}
                counts["front_then_target_rear"] = sum(r["_front_then_target"] for r in rows)
                counts["episodes_with_duplicate_requests"] = sum(r["_duplicate_requests"] > 0 for r in rows)
                counts["selection_complete"] = sum(r["selection_complete"] for r in rows)
                counts["unsupported_decisions"] = sum(r["multistep_evaluation"]["unsupported_decision"] for r in rows)
                means = {key: sum(r[key] for r in rows) / n for key in
                         ("requests", "input_tokens", "output_tokens", "wall_latency_s", "model_calls", "new_image_requests")}
                means["duplicate_requests"] = sum(r["_duplicate_requests"] for r in rows) / n
                accounting = all(r["token_accounting_complete"] for r in rows)
                if not accounting:
                    means["input_tokens"] = means["output_tokens"] = None
                detail = {}
                if track == "assisted_finite_choice":
                    assessable = sum(r["front_slot_assessable"] for r in rows)
                    detail = {
                        "front_slot_assessable_episodes": assessable,
                        "selected_chain_target_correct_pct_given_assessable": (
                            100 * counts["selected_chain_target_correct"] / assessable if assessable else None),
                        "assisted_scoring_raw_counts": {
                            key: sum(r["multistep_evaluation"][key] for r in rows)
                            for key in ("chain_grounded_correct", "localized_grounded_correct")},
                        "assisted_scoring_scope": "Publicly generated region candidates and automatic identity links; not native grounding performance"}
                else:
                    found = [r for r in rows if r["multistep_evaluation"]["procedural_evidence_sufficient"]]
                    definite = [r for r in rows if r["_definite_finished"]]
                    links_valid = sum(r["_identity_links_valid"] for r in definite)
                    detail = {
                        "decision_when_evidence_found": {
                            "evidence_found_episodes": len(found),
                            "source_label_correct": sum(r["multistep_evaluation"]["source_label_correct"] for r in found),
                            "source_label_accuracy_pct": (
                                100 * sum(r["multistep_evaluation"]["source_label_correct"] for r in found) / len(found)
                                if found else None),
                            "scope": "Conditional source-label agreement, not reliable grounded correctness"},
                        "identity_links": {
                            "definite_finished_episodes": len(definite),
                            "valid_complete_identity_chains": links_valid,
                            "valid_pct_given_definite_finish": 100 * links_valid / len(definite) if definite else None}}
                entry["policies"][policy] = {
                    **detail, "episodes": n, "privileged_reference": policy == "oracle", "counts": counts,
                    "rates_pct": {key: 100 * value / n for key, value in counts.items()},
                    "mean": means, "token_accounting_complete": accounting,
                    "paired_history_behavior": _paired_behavior(records, cases, groups, policy)}
            entry["adaptive_minus_baseline"] = {}
            case_groups = {k: c["paired_history_id"] for k, c in cases.items()}
            for baseline in ("fixed_restoration", "fixed_view_order", "random_request"):
                entry["adaptive_minus_baseline"][baseline] = {
                    key: bank_bootstrap({
                        case_id: int(_metric(records[(case_id, "adaptive_acquisition")], key))
                                 - int(_metric(records[(case_id, baseline)], key)) for case_id in cases},
                        case_groups, seed=seed, replicates=replicates) for key in metrics}
        report["models"][name] = entry
    return report


def markdown(report):
    lines = ["# 多步程序化搜证实验", "",
             "仅报告程序化仿真；不是独立人审或真实照片的证据充分性结果。", ""]
    if not report["complete"]:
        lines += ["实验尚未全部完整；未完整模型仅列完成数量，完整模型单独报告。暂不做跨模型比较。", "",
                  "| 模型 | 已完成 | 计划 |", "| --- | ---: | ---: |"]
        for name, entry in report["models"].items():
            lines.append(f"| {name} | {entry['completed_episodes']} | {entry['expected_episodes']} |")
        if not any(entry["complete"] for entry in report["models"].values()):
            return "\n".join(lines) + "\n"
        lines.append("")
    assisted = report["interface_track"] == "assisted_finite_choice"
    metrics = (("procedural_evidence_sufficient", "source_label_correct", "choice_valid",
                "selected_chain_target_correct", "finished") if assisted else METRICS)
    lines += [f"每策略 {report['cohort']['episodes_per_policy']} 个案件；{report['cohort']['paired_banks']} 个配对场景作为统计单位。",
              "搜索、提交失败和弃答都保留在分母；oracle为特权参考。", ""]
    if assisted:
        lines += ["此表为有界面辅助的有限选项诊断；候选区域和身份链接由程序生成，不代表模型原生grounding能力。",
                  "", "| 模型 | 策略 | 找到证据% | 来源标签正确% | 选择有效% | 候选链目标正确%(全部) | 完成提交% | 请求 | 重复请求 | 输入token | 输出token | 秒 |"]
    else:
        lines += ["| 模型 | 策略 | 找到证据% | 证据链正确% | 区域定位正确% | 来源标签正确% | 完成提交% | 请求 | 重复请求 | 输入token | 输出token | 秒 |"]
    lines += ["| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    labels = {"adaptive_acquisition": "自适应", "fixed_restoration": "固定恢复原图",
              "fixed_view_order": "固定视角顺序", "random_request": "随机请求", "oracle": "特权oracle"}
    def number(value):
        return "—" if value is None else f"{value:.1f}"
    for name, model in report["models"].items():
        if not model["complete"]:
            continue
        for policy, values in model["policies"].items():
            rates, means = values["rates_pct"], values["mean"]
            cells = [rates[k] for k in metrics]
            cells += [means[k] for k in ("requests", "duplicate_requests", "input_tokens", "output_tokens", "wall_latency_s")]
            lines.append("| " + " | ".join([name, labels[policy], *map(number, cells)]) + " |")
    outcome = "来源标签正确率" if assisted else "证据链正确率"
    lines += ["", f"配对95%置信区间按整个场景重采样；以下为自适应减去固定/随机策略的{outcome}差（百分点）。",
              "", "| 模型 | 对照 | 差值 | 95%区间 |", "| --- | --- | ---: | --- |"]
    for name, model in report["models"].items():
        if not model["complete"]:
            continue
        for baseline, values in model["adaptive_minus_baseline"].items():
            result = values["source_label_correct" if assisted else "chain_grounded_correct"]
            low, high = result["paired_bank_bootstrap_ci95_pp"]
            lines.append(f"| {name} | {labels[baseline]} | {result['difference_pp']:.1f} | [{low:.1f}, {high:.1f}] |")
    lines += ["", "“找到证据”由实际观察的有限world兼容集合判定。"]
    if not assisted:
        lines += ["“证据链正确”还要求判断和身份引用正确；“区域定位正确”进一步要求诊断框定位。",
                  "JSON另列拿到证据后的条件标签正确率及确定提交的身份链接有效率；它们不是可靠grounded分数。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="MODEL=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        name, separator, path = value.partition("=")
        if not separator or not name or name in runs:
            parser.error("Use unique MODEL=DIR run specifications")
        runs[name] = Path(path)
    report = analyze(args.data, runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "multistep_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "multistep_analysis.zh-CN.md").write_text(markdown(report))
    print(json.dumps({"complete": report["complete"],
                      "models": {k: {"completed": v["completed_episodes"], "expected": v["expected_episodes"]}
                                 for k, v in report["models"].items()}}))


if __name__ == "__main__":
    main()

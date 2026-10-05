"""Source-family intervals and integrity checks for paired multi-model evaluation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

VARIANTS = ["Sufficient", "Obtainable", "Missing", "Unavailable"]


def _ci95(values, iterations=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(42)
    samples = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def interval(values, iterations=10000):
    """Public statistics API: one value per source family."""
    return {"families": len(values), "mean": float(np.mean(values)) if len(values) else None,
            "ci95": _ci95(values, iterations)}


def summarize(directory, allow_partial=False):
    """Final run API rejects incomplete evaluation by default."""
    return model_summary(Path(directory), allow_partial)


def summarize_rows(rows):
    if not rows:
        return None
    families = defaultdict(list)
    for row in rows:
        families[row["family_id"]].append(row)
    answer = {"episodes": len(rows), "source_families": len(families)}
    for key in ["correct", "grounded_correct", "unsupported_decision", "coverage"]:
        values = [statistics.mean(r[key] for r in group) for group in families.values()]
        answer[key] = {"mean": statistics.mean(r[key] for r in rows), "family_ci95": _ci95(values)}
    determinate = [r for r in rows if r["coverage"]]
    answer["selective_error"] = statistics.mean(not r["correct"] for r in determinate) if determinate else None
    answer["completed"] = sum(r["termination"] == "finished" for r in rows)
    answer["terminations"] = dict(Counter(r["termination"] for r in rows))
    answer["errors"] = sum(r["errors"] for r in rows)
    for key in ["requests", "tool_calls", "request_cost", "tool_cost", "input_tokens", "output_tokens", "latency_s", "wall_latency_s", "model_calls"]:
        answer["mean_" + key] = statistics.mean(r[key] for r in rows)
    answer["max_peak_memory_gb"] = max(r["peak_memory_gb"] for r in rows)
    answer["tools"] = dict(Counter(name for r in rows for name in r["calls"]))
    return answer


def paired(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["case_id"]][row["policy"]] = row
    differences = defaultdict(list)
    for pair in grouped.values():
        if set(pair) != {"initial", "agent"}:
            continue
        differences[pair["agent"]["family_id"]].append(
            int(pair["agent"]["grounded_correct"]) - int(pair["initial"]["grounded_correct"]))
    values = [statistics.mean(group) for group in differences.values()]
    return {"grounded_proxy_delta": statistics.mean(values), "family_ci95": _ci95(values),
            "paired_source_families": len(values)} if values else None


def model_summary(directory, allow_partial=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    config = manifest["configuration"]
    expected_families = config.get("selected_families") or config["limit_families"] or config["dataset_manifest"]["families"]
    expected_episodes = expected_families * 4 * 2
    keys = [(r["case_id"], r["policy"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate episode results")
    complete = len(rows) == expected_episodes
    if not complete and not allow_partial:
        raise ValueError(f"incomplete run: {directory} ({len(rows)}/{expected_episodes})")
    if complete:
        per_family = defaultdict(set)
        for row in rows:
            per_family[row["family_id"]].add((row["variant"], row["policy"]))
        expected = {(v, p) for v in VARIANTS for p in ("initial", "agent")}
        if len(per_family) != expected_families or any(group != expected for group in per_family.values()):
            raise ValueError("missing condition or policy in a completed family")
    result = {"model": config["model_id"], "complete": complete, "expected_episodes": expected_episodes,
        "actual_episodes": len(rows), "source_families": len({r["family_id"] for r in rows}),
        "source_commit": manifest["source_commit"], "dataset_sha256": config["dataset_sha256"],
        "revision": config["revision"], "worker": manifest.get("worker"),
        "dataset_manifest":config.get("dataset_manifest",{}),
        "overall": {p: summarize_rows([r for r in rows if r["policy"] == p]) for p in ("initial", "agent")},
        "by_variant": {}, "by_category": {}}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        result["by_variant"][variant] = {p: summarize_rows([r for r in selected if r["policy"] == p]) for p in ("initial", "agent")}
        result["by_variant"][variant]["paired"] = paired(selected)
    for category in sorted({r["category"] for r in rows}):
        selected = [r for r in rows if r["category"] == category]
        result["by_category"][category] = {p: summarize_rows([r for r in selected if r["policy"] == p]) for p in ("initial", "agent")}
    return result



def merge_shards(directories, destination):
    """Join disjoint complete runs of one model; preserve the original manifests."""
    manifests = [json.loads((p / "manifest.json").read_text()) for p in directories]
    first = manifests[0]
    shards = first["configuration"]["shards"]
    if {m["configuration"]["shard_index"] for m in manifests} != set(range(shards)):
        raise ValueError("all execution shards are required")
    def common(m):
        return {k:v for k,v in m["configuration"].items() if k not in {"shard_index","selected_families"}}
    if any(common(m) != common(first) for m in manifests):
        raise ValueError("shard configurations differ")
    for path in directories:
        model_summary(path)
    rows = [json.loads(line) for path in directories for line in (path / "results.jsonl").read_text().splitlines()]
    if len({(r["case_id"],r["policy"]) for r in rows}) != len(rows):
        raise ValueError("overlapping shard episodes")
    manifest = dict(first)
    manifest["configuration"] = {**first["configuration"], "shard_index": None,
                                 "selected_families": len({r["family_id"] for r in rows})}
    manifest["execution_shards"] = manifests
    manifest["worker"] = {"workers": [m.get("worker") for m in manifests]}
    destination.mkdir(parents=True,exist_ok=True)
    if (destination / "results.jsonl").exists():
        raise ValueError("merged results already exist")
    (destination / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (destination / "results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in sorted(rows,key=lambda r:(r["case_id"],r["policy"]))))
    model_summary(destination)


def markdown(report):
    dataset=report["models"][0].get("dataset_manifest",{})
    chain=dataset.get("grounding_protocol")=="evidence-chain-v1"
    lines=["# 冻结公共数据的多模型评测", "",
        f"数据：{dataset.get('dataset','见运行清单')}；统计单位：原始案件族／序列，四种证据版本共同重采样。",
        "候选标注尚未独立审核时，证据充分性结果属于协议代理指标，不能称为正式 grounded accuracy。", "",
        "| 模型 | 证据版本 | 初始池评分 | Agent 评分 | 配对变化（95% 案件族 CI） | 平均索证次数 | 完成 Agent 案件 |",
        "|---|---|---:|---:|---|---:|---:|"]
    for model in report["models"]:
        for variant, data in model["by_variant"].items():
            if not data["initial"] or not data["agent"] or not data["paired"]:
                continue
            delta = data["paired"]
            lo, hi = delta["family_ci95"]
            lines.append(f"| {model['model']} | {variant} | {data['initial']['grounded_correct']['mean']:.1%} | {data['agent']['grounded_correct']['mean']:.1%} | {delta['grounded_proxy_delta']:+.1%} [{lo:+.1%}, {hi:+.1%}] | {data['agent']['mean_requests']:.2f} | {data['agent']['completed']}/{data['agent']['episodes']} |")
    note=("证据链协议区分物理真假和可获得的充分证据：必要材料不可获得时，Need more evidence 可以是正确案件目标。"
          if chain else "旧图片集合代理使用二分类视觉标签，安全弃答不等于标签正确；Missing/Unavailable 的零分需要结合无依据结论、覆盖率和完成率解读。")
    lines += ["", note, "",
        "| 模型 | Agent 覆盖率 | 无依据确定结论 | 选择性错误率 | 工具费用 | 索证费用 | 案件耗时（秒） | 格式／工具错误 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in report["models"]:
        a = model["overall"]["agent"]
        if a:
            risk = f"{a['selective_error']:.1%}" if a['selective_error'] is not None else 'n/a'
            lines.append(f"| {model['model']} | {a['coverage']['mean']:.1%} | {a['unsupported_decision']['mean']:.1%} | {risk} | {a['mean_tool_cost']:.2f} | {a['mean_request_cost']:.2f} | {a['mean_wall_latency_s']:.2f} | {a['errors']} |")
    lines += ["", "状态："+("全部完成。" if report["complete"] else "部分结果，评测尚未完成。"), "",
        "置信区间以原始案件族／序列为单位，保留相关证据版本。初始池策略只看到初始材料；Agent 使用相同冻结工具与预算。"
        "各模型使用相同 portable 工具传输与确定性生成。未完成案件计零分，保留实际终止原因。", "",
        "这些结果不证明训练增益、仿真到真实的迁移或真实前后损坏变化。候选身份标注来自公开序列，充分性仍需独立审核。"
        "模型版本、数据哈希、协议、预算、分项统计和用量见配套 JSON；研究边界见 [中文协议](../research_protocol.zh-CN.md)。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if args.merge_shards:
        merge_shards(args.runs, args.output)
        return
    models = [model_summary(path, args.allow_partial) for path in args.runs]
    if len({m["dataset_sha256"] for m in models}) != 1:
        raise ValueError("models evaluated different frozen datasets")
    report = {"complete": all(m["complete"] for m in models), "models": models,
              "bootstrap": "10000 source-family resamples; seed 42", "metric": "source-derived evidence-chain proxy" if models[0].get("dataset_manifest",{}).get("grounding_protocol")=="evidence-chain-v1" else "original-image quality proxy"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"complete": report["complete"], "episodes": sum(m["actual_episodes"] for m in models)}), flush=True)


if __name__ == "__main__":
    main()

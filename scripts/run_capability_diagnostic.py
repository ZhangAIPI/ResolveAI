"""Run paired development diagnostics on four already allocated GPUs."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from resolveai.diagnostic import original_id, request_query, select_cases
from resolveai.environment import Environment

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO.parent/"resolveai-poc"
OUTPUT = RUNTIME/"evaluation/capability-diagnostic-v1"
RUNS = [
    ("qwen3", "Qwen/Qwen3-VL-8B-Instruct", RUNTIME/"model", 0, 1, 0,
     "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    ("qwen25", "Qwen/Qwen2.5-VL-7B-Instruct", RUNTIME/"models/qwen25", 1, 1, 0,
     "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    ("internvl35-shard0", "OpenGVLab/InternVL3_5-8B-HF", RUNTIME/"models/internvl35", 2, 2, 0,
     "741a7d03020411e666c6109218ab71e08151ef86"),
    ("internvl35-shard1", "OpenGVLab/InternVL3_5-8B-HF", RUNTIME/"models/internvl35", 3, 2, 1,
     "741a7d03020411e666c6109218ab71e08151ef86"),
]


def prepare():
    cases = select_cases(RUNTIME/"benchmark-data", RUNTIME/"co3d-candidates")
    assert len(cases) == 120
    assert Counter(c["diagnostic_truth"] for c in cases) == {"Supported":60, "Refuted":60}
    checked = 0
    for case in cases:
        env = Environment(case, case["asset_root"])
        state = env._world.fingerprint
        result = env.step({"type":"request_photo", "query":request_query(case)})
        if result.get("status") != "provided" or original_id(case) not in env._released:
            raise RuntimeError("benchmark retrieval preflight failed")
        assert state == env._world.fingerprint
        checked += 1
    plan = {"protocol":"capability-diagnostic-v1", "development_only":True,
        "scope":"public candidate labels; target regions supplied; no reviewed sufficiency claim",
        "families":len(cases), "source_counts":dict(Counter(c["diagnostic_source"] for c in cases)),
        "class_counts":dict(Counter(c["diagnostic_truth"] for c in cases)),
        "identity_controls":"30 same-sequence positive / 30 easy cross-category negative; no hard-negative claim",
        "source_reuse":"Generated identity controls share source sequences; no independent-family confidence intervals.",
        "preflight":{"scripted_request_passed":checked, "world_state_unchanged":checked},
        "data_sha256":{name:hashlib.sha256((RUNTIME/name/"cases.json").read_bytes()).hexdigest()
                       for name in ["benchmark-data","co3d-candidates"]},
        "cases":cases}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"images").mkdir(exist_ok=True)
    path=OUTPUT/"plan.json"
    serialized=json.dumps(plan,ensure_ascii=False,indent=2)+"\n"
    if path.exists() and path.read_text()!=serialized:
        raise ValueError("diagnostic plan changed; use a new experiment")
    path.write_text(serialized)
    return path


def worker(spec, plan):
    name, model_id, model, device, shards, shard_index, revision=spec
    output=OUTPUT/name
    output.mkdir(exist_ok=True)
    images=output/"images"
    if not images.exists():
        images.symlink_to("../images",target_is_directory=True)
    command=[sys.executable,"-m","resolveai.diagnostic",str(plan),str(model),str(output),
             "--model-id",model_id,"--revision",revision,"--device",str(device),
             "--shards",str(shards),"--shard-index",str(shard_index)]
    with (output/"worker.log").open("a") as log:
        subprocess.run(command,cwd=REPO,stdout=log,stderr=log,check=True)
    return name


def publish(plan_path):
    plan=json.loads(plan_path.read_text())
    models=defaultdict(list)
    for name,model_id,*_ in RUNS:
        models[model_id].extend(json.loads(line) for line in (OUTPUT/name/"results.jsonl").read_text().splitlines())
    arms=["initial","scripted_request","menu_agent","rich_agent","menu_unavailable","rich_unavailable"]
    expected={(c["family_id"],arm) for c in plan["cases"] for arm in arms}
    report={"protocol":plan["protocol"],"complete":True,"source_commit":subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "preflight":plan["preflight"],"families":120,"episodes":2160,
        "limitations":[plan["scope"],plan["source_reuse"],plan["identity_controls"],
            "Finite-action menus control tool arguments and decoding; not free-form agent equivalence.",
            "Correct supplied-source labels are not independently reviewed visual sufficiency.",
            "Unavailable conclusions are described, not assigned a synthetic evidence-sufficiency label."],
        "models":{}}
    text=["# 模型能否搜证：同题对照诊断\n",
        "这次把“看图判断”“选择搜证动作”“填写工具参数”分开。120 个基础题目，"
        "60 个状态题与 60 个身份题，支持/反驳各半；三个模型各跑六种条件，共 2,160 次。\n",
        "身份目标区域公开给定以排除物体指代与定位歧义；身份负例是不同类别的容易控制题，"
        "不是细粒度身份硬负例。标注仍来自公共数据。原数据图片未改动。\n",
        f"环境自检：{plan['preflight']['scripted_request_passed']}/120 次指定请求取得正确原图，世界状态保持不变。\n"]
    for model,rows in models.items():
        keys=[(r["family_id"],r["arm"]) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:
            raise ValueError("incomplete or overlapping diagnostic results")
        by_family=defaultdict(dict)
        for row in rows:by_family[row["family_id"]][row["arm"]]=row
        grouped={}
        text.extend([f"\n## {model}\n",
            "| 题目 | 初始图答对 | 直接补齐后答对 | 菜单搜证后答对 | 完整接口搜证后答对 |\n",
            "|---|---:|---:|---:|---:|\n"])
        for source in ["mvtec","co3d"]:
            families=[v for v in by_family.values() if v["initial"]["source"]==source]
            metrics={a:sum(f[a]["label_correct"] for f in families) for a in arms[:4]}
            sensitive=[f for f in families if f["scripted_request"]["label_correct"] and not f["initial"]["label_correct"]]
            conditional={}
            for a in ["menu_agent","rich_agent"]:
                conditional[a]={
                    "answered_correctly":sum(f[a]["label_correct"] for f in sensitive),
                    "obtained_original_and_correct":sum(f[a]["received_target_original"] and f[a]["label_correct"] for f in sensitive),
                    "did_not_obtain_original":sum(not f[a]["received_target_original"] for f in sensitive),
                    "obtained_original_but_wrong":sum(f[a]["received_target_original"] and not f[a]["label_correct"] for f in sensitive),
                    "all_cases_finished":sum(f[a]["finished"] for f in families),
                    "all_cases_errors":sum(f[a]["errors"] for f in families)}
            unavailable={a:{
                "finished_need_more":sum(f[a]["finished"] and f[a]["verdict"]=="Need more evidence" for f in families),
                "determinate":sum(f[a]["verdict"] in {"Supported","Refuted"} for f in families),
                "failed":sum(not f[a]["finished"] for f in families)}
                for a in ["menu_unavailable","rich_unavailable"]}
            grouped[source]={"cases":len(families),"correct_by_arm":metrics,
                "evidence_sensitive_cases":len(sensitive),"conditional":conditional,
                "unavailable_descriptive":unavailable}
            label="物品状态" if source=="mvtec" else "跨图身份"
            text.append("| "+label+" | "+" | ".join(f"{metrics[a]}/{len(families)}" for a in arms[:4])+" |\n")
            text.append("")
        for source,group in grouped.items():
            title="状态题" if source=="mvtec" else "身份题"
            n=group["evidence_sensitive_cases"]
            text.append(f"\n{title}中，{n} 道题初始答错、直接补齐后答对。这些是本轮可诊断搜证的候选题。\n")
            for a,label in [("menu_agent","菜单"),("rich_agent","完整接口")]:
                c=group["conditional"][a]
                text.append(f"- {label}：最终 {c['answered_correctly']}/{n} 答对；"
                    f"{c['obtained_original_and_correct']} 题取得原图且答对；"
                    f"{c['did_not_obtain_original']} 题未取得原图；"
                    f"{c['obtained_original_but_wrong']} 题取得后仍答错。\n")
        report["models"][model]=grouped
    text.extend(["\n## 如何据此决策\n",
        "- 简单菜单成功、完整接口失败：先修工具接口与参数生成，不直接称为视觉能力不足。\n",
        "- 补齐材料能答对，但菜单下不取证且仍答错：这是搜索策略训练的候选失败案例，仍需复核题目。\n",
        "- 取证后仍答错：检查多轮历史处理与证据整合。\n",
        "- 补齐材料也答错：先人工复核题目、来源标签与视觉可判断性，再区分感知能力不足。\n",
        "- 初始图片就能答对：该题不能用来证明索证收益；预览不能机械判为不充分。\n",
        "\n菜单采用模型对可见操作选项的下一 token 分数选择，免除 JSON 与坐标生成；"
        "它是诊断支架，不代表模型已经学会自由工具调用。完整接口沿用六回合、预算 12 的原生调用。"
        "不可获取条件只描述行为，不把缺原图等同于视觉上无法判断。共享来源不报告独立案件置信区间。\n"])
    for suffix,content in [("json",json.dumps(report,ensure_ascii=False,indent=2)+"\n"),("md","".join(text))]:
        (OUTPUT/("summary."+suffix)).write_text(content)
        (REPO/("docs/results/capability_diagnostic."+suffix)).write_text(content)
    subprocess.run(["git","diff","--check"],cwd=REPO,check=True)
    subprocess.run(["git","add","--","docs/results/capability_diagnostic.md","docs/results/capability_diagnostic.json"],cwd=REPO,check=True)
    subprocess.run(["git","commit","--only","docs/results/capability_diagnostic.md",
        "docs/results/capability_diagnostic.json","-m","Report paired perception, acquisition and interface diagnostics"],cwd=REPO,check=True)
    env=os.environ.copy()
    env["GIT_SSH_COMMAND"]="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"
    subprocess.run(["git","push","origin","main"],cwd=REPO,env=env,check=True)


if __name__=="__main__":
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Run inside the allocated GPU node, never on the jump host")
    plan=prepare()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda spec:worker(spec,plan),RUNS):
            print(json.dumps({"completed_worker":result}),flush=True)
    publish(plan)
    print(json.dumps({"complete":True,"episodes":2160}),flush=True)

"""Balanced, reproducible human study with source-disjoint participant tasks."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import secrets

ARMS = ("initial", "interactive", "full_available")
VARIANTS = ("Sufficient", "Obtainable", "Missing", "Unavailable")


def balanced_pick(groups, count, rng):
    buckets = {k: list(v) for k, v in groups.items()}
    for values in buckets.values():
        rng.shuffle(values)
    keys = sorted(buckets, key=str)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < count:
        progress = False
        for key in keys:
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop())
                progress = True
        if not progress:
            raise ValueError("Insufficient distinct balanced candidates")
    return chosen


def select_families(cases, seed=20261005, *, state_pairs=24, identity_pairs=16, unknown_count=16):
    rng = random.Random(seed)
    states = defaultdict(lambda: defaultdict(list))
    identities, unknown = defaultdict(dict), defaultdict(list)
    for case in cases:
        if case["variant"] != "Obtainable":
            continue
        if case["task"] == "state":
            states[(case["category"], case["provenance"]["criterion"])][case["provenance_truth"]].append(case)
        elif case["provenance"]["construction"] == "same-category-candidate":
            unknown[case["category"]].append(case)
        else:
            identities[(case["category"], case["evidence"][0]["source_id"])][case["provenance"]["construction"]] = case
    pairs = defaultdict(list)
    for (category, criterion), labels in sorted(states.items()):
        rng.shuffle(labels["Supported"]); rng.shuffle(labels["Refuted"])
        pairs[category].extend(zip(labels["Supported"], labels["Refuted"]))
    chosen = [c for pair in balanced_pick(pairs, state_pairs, rng) for c in pair]
    controls = defaultdict(list)
    for (category, _), pair in identities.items():
        if set(pair) == {"source-positive", "easy-negative"}:
            controls[category].append((pair["source-positive"], pair["easy-negative"]))
    chosen += [c for pair in balanced_pick(controls, identity_pairs, rng) for c in pair]
    chosen += balanced_pick(unknown, unknown_count, rng)
    if len({c["family_id"] for c in chosen}) != state_pairs*2+identity_pairs*2+unknown_count:
        raise ValueError("Study requires distinct claim families")
    return chosen


def assign(families, participants=24, repeats=2, seed=20261005):
    counts = Counter(c["task"] for c in families)
    if any(n*repeats % participants for n in counts.values()):
        raise ValueError("Participant count must divide per-task assignments")
    quotas = {(task, arm): n*repeats//participants for task,n in counts.items() for arm in ARMS}
    variants, buckets = {}, defaultdict(list)
    for case in families:
        buckets[case["provenance"].get("construction", "state")].append(case)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
        for i, case in enumerate(bucket):
            variants[case["family_id"]] = VARIANTS[i % 4]
    slots = [(case, arm) for case in families for arm in ARMS for _ in range(repeats)]
    frequency = Counter(g for c,_ in slots for g in c["group_ids"])
    if max(frequency.values(),default=0)>participants:
        raise ValueError("A source is repeated too often for independent assignments")
    for attempt in range(1000):
        rng=random.Random(seed+attempt)
        queues=[[] for _ in range(participants)]
        used=[set() for _ in queues]; remaining=[dict(quotas) for _ in queues]
        shuffled=list(slots); rng.shuffle(shuffled)
        shuffled.sort(key=lambda slot:-max(frequency[g] for g in slot[0]["group_ids"]))
        for case,arm in shuffled:
            key=(case["task"],arm)
            eligible=[p for p in range(participants) if remaining[p][key]>0
                      and not used[p].intersection(case["group_ids"])]
            if not eligible:
                break
            rng.shuffle(eligible)
            p=max(eligible,key=lambda p:remaining[p][key])
            variant=variants[case["family_id"]]
            queues[p].append({"family_id":case["family_id"],"case_id":case["family_id"]+"-"+variant,"condition":arm})
            used[p].update(case["group_ids"]); remaining[p][key]-=1
        else:
            for queue in queues: rng.shuffle(queue)
            return queues,variants
    raise ValueError("No source-disjoint schedule; revise the sample before recruiting")


def prepare(data, output):
    if output.exists(): raise ValueError("Use a fresh study directory")
    families=select_families(json.loads((data/"cases.json").read_text()))
    queues,variants=assign(families)
    plan={"protocol":"human-study-v1","seed":20261005,"data":str(data.resolve()),
          "dataset_sha256":hashlib.sha256((data/"cases.json").read_bytes()).hexdigest(),
          "family_ids":sorted(c["family_id"] for c in families),"human_case_variants":variants,
          "participants":24,"tasks_per_participant":24,"conditions":list(ARMS),"budget":12,"max_turns":8,
          "source_disjoint_within_participant":True,
          "scope":"Public-data development validation; source labels are not reviewed visual truth"}
    access={}
    for i,queue in enumerate(queues,1):
        access[secrets.token_urlsafe(24)]={"id":f"P{i:02d}","role":"participant","tasks":queue}
    for i in range(1,3):
        order=list(plan["family_ids"]); random.Random(20261005+i).shuffle(order)
        access[secrets.token_urlsafe(24)]={"id":f"R{i:02d}","role":"reviewer","families":order}
    output.mkdir(parents=True)
    (output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    (output/"access.json").write_text(json.dumps(access,indent=2)+"\n")
    (output/"access.json").chmod(0o600)
    return plan


def short_assign(families, seed=20261005):
    """20 people: one trial per acquisition arm and two blind review families."""
    rng = random.Random(seed)
    state_pairs = defaultdict(dict)
    for case in families:
        if case["task"] == "state":
            state_pairs[(case["category"], case["provenance"]["criterion"])][case["provenance_truth"]] = case
    keys = list(state_pairs); rng.shuffle(keys)
    searching = [c for key in keys[:3] for c in state_pairs[key].values()]
    controls = defaultdict(dict)
    unknown = []
    for case in families:
        if case["task"] != "identity": continue
        kind = case["provenance"]["construction"]
        if kind == "same-category-candidate": unknown.append(case)
        else: controls[case["evidence"][0]["source_id"]][kind] = case
    key = sorted(controls)[rng.randrange(len(controls))]
    searching += list(controls[key].values()) + unknown
    if len(searching) != 10: raise ValueError("Expected ten search families")
    slots = [(c,arm) for c in searching for arm in ARMS for _ in range(2)]
    for attempt in range(2000):
        rng = random.Random(seed+attempt)
        queues=[[] for _ in range(20)];used=[set() for _ in queues];counts=[Counter() for _ in queues]
        shuffled=list(slots);rng.shuffle(shuffled)
        for case,arm in shuffled:
            eligible=[p for p in range(20) if counts[p][arm]==0 and counts[p][case["task"]]<2
                      and not used[p].intersection(case["group_ids"])]
            if not eligible: break
            rng.shuffle(eligible);person=min(eligible,key=lambda p:len(queues[p]))
            queues[person].append({"mode":"search","family_id":case["family_id"],"condition":arm})
            counts[person][arm]+=1;counts[person][case["task"]]+=1
            used[person].update(case["group_ids"])
        else:
            reviews=[c for c in families for _ in range(2)];rng.shuffle(reviews)
            reviews.sort(key=lambda c:-sum(bool(set(c["group_ids"])&u) for u in used))
            review_counts=[0]*20
            for case in reviews:
                eligible=[p for p in range(20) if review_counts[p]<2
                          and not used[p].intersection(case["group_ids"])]
                if not eligible: break
                rng.shuffle(eligible);person=min(eligible,key=lambda p:review_counts[p])
                queues[person].append({"mode":"review","family_id":case["family_id"]})
                used[person].update(case["group_ids"]);review_counts[person]+=1
            else:
                variants={c["family_id"]:VARIANTS[i%4] for i,c in enumerate(sorted(searching,key=lambda c:c["family_id"]))}
                for queue in queues:
                    searching_rows=queue[:3];rng.shuffle(searching_rows)
                    queue[:3]=searching_rows
                    for row in searching_rows: row["case_id"]=row["family_id"]+"-"+variants[row["family_id"]]
                return queues,variants
    raise ValueError("Could not assign short study without source overlap")


def prepare_short(data, output):
    if output.exists(): raise ValueError("Use a fresh short-study directory")
    families=select_families(json.loads((data/"cases.json").read_text()),state_pairs=5,identity_pairs=4,unknown_count=2)
    queues,variants=short_assign(families)
    plan={"protocol":"human-study-short-v2","seed":20261005,"data":str(data.resolve()),
          "dataset_sha256":hashlib.sha256((data/"cases.json").read_bytes()).hexdigest(),
          "family_ids":sorted(c["family_id"] for c in families),"human_case_variants":variants,
          "participants":20,"tasks_per_participant":5,"search_tasks_per_participant":3,
          "review_tasks_per_participant":2,"conditions":list(ARMS),"budget":12,"max_turns":8,
          "search_seconds":75,"review_seconds":120,"source_disjoint_within_participant":True,
          "scope":"20-person pilot sampled from the whole public pool; no population-level power claim"}
    access={secrets.token_urlsafe(24):{"id":f"P{i:02d}","role":"participant","tasks":queue}
            for i,queue in enumerate(queues,1)}
    access[secrets.token_urlsafe(32)]={"id":"organizer","role":"admin"}
    output.mkdir(parents=True)
    (output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    (output/"access.json").write_text(json.dumps(access,indent=2)+"\n")
    (output/"access.json").chmod(0o600)
    return plan

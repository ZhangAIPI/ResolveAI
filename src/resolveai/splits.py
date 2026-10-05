"""Deterministic connected-component splits to prevent asset/family leakage."""
import hashlib


def split_families(families, seed="resolveai-v1"):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for family in families:
        keys = ["family:" + family["family_id"]]
        keys += ["asset:" + a for a in family["group_ids"]]
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
    components = {}
    for key in list(parent):
        components.setdefault(find(key), []).append(key)
    labels = {}
    for representative, keys in components.items():
        digest = hashlib.sha256((seed + "|" + "|".join(sorted(keys))).encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        labels[representative] = "train" if fraction < .7 else "val" if fraction < .8 else "test"
    return {f["family_id"]: labels[find("family:" + f["family_id"])] for f in families}

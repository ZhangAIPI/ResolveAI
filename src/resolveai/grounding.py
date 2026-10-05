"""Private region and evidence-chain evaluation, independent of model observation text."""
from .tools import RELATIONS


def covers(required, cited_boxes):
    """Exact union coverage: repeated/overlapping crops cannot inflate evidence."""
    x0, y0, x1, y1 = required
    boxes = [[max(x0, a), max(y0, b), min(x1, c), min(y1, d)] for a, b, c, d in cited_boxes]
    boxes = [b for b in boxes if b[0] < b[2] and b[1] < b[3]]
    xs = sorted({x0, x1, *[x for box in boxes for x in (box[0], box[2])]})
    area = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] <= left and b[2] >= right)
        length, end = 0, y0
        for low, high in intervals:
            length += max(0, high - max(end, low))
            end = max(end, high)
        area += (right - left) * length
    return area >= (x1 - x0) * (y1 - y0)



def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


def endpoint_matches(expected, actual, threshold):
    return (expected["image_id"] == actual["image_id"] and expected["time"] == actual["time"]
            and iou(expected["bbox"], actual["bbox"]) >= threshold)


def link_matches(expected, actual):
    if expected["relation"] != actual["relation"]:
        return False
    threshold = expected.get("min_iou", .5)
    direct = all(endpoint_matches(expected[k], actual[k], threshold) for k in ("left", "right"))
    symmetric = expected["relation"] in {"same_object", "different_object", "same_time"}
    reverse = symmetric and all(endpoint_matches(expected[a], actual[b], threshold)
                                for a, b in (("left", "right"), ("right", "left")))
    return direct or reverse


def requirement_matches(requirement, citations, links):
    if isinstance(requirement, str):
        return any(c["image_id"] == requirement for c in citations)
    if "relation" in requirement:
        return any(link_matches(requirement, link) for link in links)
    boxes = [c["bbox"] for c in citations if c["image_id"] == requirement["image_id"]
             and c["time"] == requirement["time"]]
    return covers(requirement["bbox"], boxes)


def alternatives_satisfied(alternatives, citations, links):
    return any(group and all(requirement_matches(r, citations, links) for r in group)
               for group in alternatives)


def chain_report(annotation, verdict, citations, links):
    """Private evidence audit: all conjuncts for support, one true counterexample for refutation."""
    subclaims = annotation["subclaims"]
    coverage = {s["id"]: (s["truth"] == verdict and alternatives_satisfied(
        s["minimal_evidence_sets"].get(verdict, []), citations, links)) for s in subclaims}
    expected_links = [r for s in subclaims for groups in s["minimal_evidence_sets"].values()
                      for group in groups for r in group if isinstance(r, dict) and "relation" in r]
    valid_links = sum(any(link_matches(expected, link) for expected in expected_links) for link in links)
    sufficient = (all(coverage.values()) if verdict == "Supported" else
                  any(coverage.values()) if verdict == "Refuted" else False)
    return {"sufficient": sufficient and valid_links == len(links),
            "subclaim_coverage": coverage, "valid_links": valid_links, "predicted_links": len(links)}


def validate_annotation(annotation):
    """Reject contradictory chain labels before any agent episode starts."""
    if annotation.get("protocol") != "evidence-chain-v1":
        return
    subclaims = annotation.get("subclaims", [])
    if not subclaims or len({s["id"] for s in subclaims}) != len(subclaims):
        raise ValueError("chain requires unique nonempty subclaims")
    truths = [s["truth"] for s in subclaims]
    if any(t not in {"Supported", "Refuted", "uncertain"} for t in truths):
        raise ValueError("invalid subclaim truth")
    derived = "Refuted" if "Refuted" in truths else "Supported" if all(t == "Supported" for t in truths) else "Need more evidence"
    if annotation["verdict"] != derived:
        raise ValueError("conjunction label disagrees with subclaim truths")
    for subclaim in subclaims:
        truth = subclaim["truth"]
        if truth != "uncertain" and not subclaim["minimal_evidence_sets"].get(truth):
            raise ValueError("determinate subclaim requires independent evidence annotation")
        for groups in subclaim["minimal_evidence_sets"].values():
            for group in groups:
                if not group or any(isinstance(r, str) for r in group):
                    raise ValueError("chain requires source regions or explicit relations")
                for requirement in group:
                    endpoints = [requirement]
                    if "relation" in requirement:
                        if requirement["relation"] not in RELATIONS or not 0 < requirement.get("min_iou", .5) <= 1:
                            raise ValueError("invalid annotated relation or IoU threshold")
                        endpoints = [requirement["left"], requirement["right"]]
                        if endpoints[0]["image_id"] == endpoints[1]["image_id"]:
                            raise ValueError("annotated relation must connect distinct images")
                    for endpoint in endpoints:
                        box = endpoint["bbox"]
                        if (not endpoint["image_id"] or not endpoint["time"] or len(box) != 4
                                or any(type(x) is not int for x in box)
                                or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
                            raise ValueError("invalid annotated source region")


def sufficient(annotation, verdict, citations, links=()):
    if annotation.get("protocol") == "evidence-chain-v1":
        return chain_report(annotation, verdict, citations, links)["sufficient"]
    return alternatives_satisfied(annotation["minimal_evidence_sets"].get(verdict, []), citations, links)

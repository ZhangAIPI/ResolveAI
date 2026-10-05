"""Private region-evidence evaluation, independent of model observation text."""


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


def sufficient(annotation, verdict, citations):
    """Alternative conjunctions support full subclaims or decisive refutation.

    Legacy string image sets remain a declared image-level proxy. Structured
    requirements contain image_id, source bbox and time, from independent labels.
    """
    alternatives = annotation["minimal_evidence_sets"].get(verdict, [])
    for required_set in alternatives:
        if not required_set:
            continue
        satisfied = True
        for requirement in required_set:
            if isinstance(requirement, str):
                match = any(c["image_id"] == requirement for c in citations)
            else:
                boxes = [c["bbox"] for c in citations if c["image_id"] == requirement["image_id"]
                         and c["time"] == requirement["time"]]
                match = covers(requirement["bbox"], boxes)
            satisfied &= match
        if satisfied:
            return True
    return False

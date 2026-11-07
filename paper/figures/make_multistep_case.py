"""Render the fixed seed-71000 procedural pair; no model results are consulted.

Usage:
  python paper/figures/make_multistep_case.py --data /path/to/cases.json \
    --png /path/to/preview.png
The front surveys use the same fixed source-pixel crop. Blue target boxes are
publication annotations only; this script never writes the dataset or inputs.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

SEED_PATH = "banks/seed-71000/"
FRONT_CROP = (85, 145, 690, 365)
BLUE = "#25618a"


def load_pair(path):
    document = json.loads(path.read_text())
    cases = document["cases"] if isinstance(document, dict) else document
    pair = [case for case in cases if case["variant"] == "Obtainable"
            and any(item["path"].startswith(SEED_PATH) for item in case["evidence"])]
    pair.sort(key=lambda case: ("rear_left", "rear_center", "rear_right").index(
        case["multistep_certificate"]["target_rear_view"]))
    if len(pair) != 2 or pair[0]["paired_history_id"] != pair[1]["paired_history_id"]:
        raise ValueError("Expected the two frozen seed-71000 paired cases")
    for view in ("reference", "initial"):
        if asset(pair[0], view).read_bytes() != asset(pair[1], view).read_bytes():
            raise ValueError("The paired initial photographs must be byte-identical")
    return pair


def asset(case, view):
    item = next(item for item in case["evidence"] if item["id"] == view)
    return Path(case["asset_root"]) / item["path"]


def photograph(fig, case, view, bounds, *, crop=None, mark_target=False):
    ax = fig.add_axes(bounds)
    with Image.open(asset(case, view)) as source:
        image = source.convert("RGB")
        if crop is not None:
            image = image.crop(crop)
        ax.imshow(image, interpolation="nearest", rasterized=True)
    ax.set_axis_off()
    if mark_target:
        x0, y0, x1, y1 = case["multistep_regions"]["front_target"]["bbox"]
        left, top = crop[:2] if crop else (0, 0)
        ax.add_patch(Rectangle(
            (x0-left-.5, y0-top-.5), x1-x0, y1-y0,
            fill=False, linewidth=.9, edgecolor=BLUE))
    return ax


def arrow(fig, start, end):
    fig.add_artist(FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=8, linewidth=.8, color="#333333",
        shrinkA=0, shrinkB=0))


def make_figure(pair, output, png):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .6, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.0, 2.8), facecolor="white")
    text = lambda x, y, label, **kw: fig.text(
        x, y, label, ha="center", va="center", **kw)
    text(.12, .973, "Shared initial views", weight="bold", fontsize=9)
    text(.53, .973, "1. Request front view", weight="bold", fontsize=9)
    text(.91, .973, "2. Choose rear", weight="bold", fontsize=9)

    photograph(fig, pair[0], "reference", [.05, .52, .15, .375])
    photograph(fig, pair[0], "initial", [.015, .095, .215, .358])
    text(.125, .925, "Reference")
    text(.1225, .48, "Occluded overview")

    rows = [(.55, .925, .525), (.12, .495, .095)]
    for index, (case, (bottom, heading, outcome_y)) in enumerate(zip(pair, rows)):
        rear = case["multistep_certificate"]["target_rear_view"]
        slot = rear.removeprefix("rear_")
        photograph(fig, case, "front_survey", [.335, bottom, .395, .35],
                   crop=FRONT_CROP, mark_target=True)
        photograph(fig, case, rear, [.835, bottom, .15, .375])
        text(.5325, heading, f"{'AB'[index]}: target at {slot}")
        text(.91, outcome_y,
             "No scratch" if case["annotation"]["verdict"] == "Refuted" else "Scratch")
        center = bottom + .175
        arrow(fig, (.737, center), (.827, center))
        text(.782, center+.047, slot, fontsize=8.5)

    # The fork represents two cases with the same initial observations.
    for points in [([.24, .28], [.50, .50]),
                   ([.28, .28], [.295, .725])]:
        fig.lines.append(plt.Line2D(*points, transform=fig.transFigure,
                                   color="#555555", linewidth=.7))
    arrow(fig, (.28, .725), (.327, .725))
    arrow(fig, (.28, .295), (.327, .295))
    text(.51, .025,
         "Cropped front views. Blue boxes are for illustration only and are not shown to models.",
         fontsize=8.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, metadata={
        "Title": "Two-step evidence acquisition on a fixed procedural case",
        "Subject": "Seed 71000 paired scenes; publication boxes are not model inputs",
        "Creator": "make_multistep_case.py",
    })
    if png:
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pdf", type=Path,
                        default=Path(__file__).with_name("multistep_case.pdf"))
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    path = args.data / "cases.json" if args.data.is_dir() else args.data
    pair = load_pair(path)
    make_figure(pair, args.pdf, args.png)
    print(json.dumps({
        "seed": 71000, "cases": [case["case_id"] for case in pair],
        "dataset_sha256": sha256(path.read_bytes()).hexdigest(),
        "source_images": {str(asset(case, view)): sha256(asset(case, view).read_bytes()).hexdigest()
                          for case in pair
                          for view in ("reference", "initial", "front_survey",
                                       case["multistep_certificate"]["target_rear_view"])},
        "front_display_crop": FRONT_CROP, "model_results_consulted": False,
        "publication_boxes_in_model_inputs": False,
        "pdf": str(args.pdf), "png": str(args.png) if args.png else None,
    }, indent=2))


if __name__ == "__main__":
    main()

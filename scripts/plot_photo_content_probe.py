"""Plot the completed matched-history probe: final responses, not benchmark accuracy."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resolveai-probe-matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Patch
import numpy as np

CONDITIONS = ("real", "gray", "shuffled")
LABELS = ("Original", "Gray", "Shuffled")
CATEGORIES = ("Supported", "Refuted")
COLORS = ("#547b93", "#414141")
CAPTION = "Responses to Changed Image Content."


def checked_counts(report):
    if (report.get("complete") is not True or report.get("histories") != 68
            or report.get("episodes") != 204
            or set(report.get("conditions", {})) != set(CONDITIONS)):
        raise ValueError("The completed 68-by-3 report is required.")
    if report.get("matching") != {
            "identical_nonpixel_public_inputs": 68,
            "identical_nonpixel_processor_tensors": 68,
            "unchanged_initial_image_blocks": 68,
            "distinct_processed_target_blocks": 68,
            "worker_token_counts_equal_preflight": 204}:
        raise ValueError("The completed input-matching audit is required.")
    if report.get("native_initial_scores_reused") is not False:
        raise ValueError("Native benchmark scores must not enter this figure.")
    counts = []
    for condition in CONDITIONS:
        group = report["conditions"][condition]
        votes = group["verdict_counts"]
        if group["episodes"] != 68 or set(votes) - {
                "Supported", "Refuted", "Need more evidence", "Invalid finish"}:
            raise ValueError("Unexpected denominator or response category.")
        if any(type(value) is not int or value < 0 for value in votes.values()):
            raise ValueError("Response counts must be nonnegative integers.")
        # This two-category figure is valid only because the completed report
        # contains no abstentions or invalid finishes. Never silently drop them.
        if (votes.get("Need more evidence", 0) or votes.get("Invalid finish", 0)
                or group["schema_valid_finish_pct"] != 100
                or group["abstained_pct"] != 0 or group["definite_pct"] != 100):
            raise ValueError("Nonzero abstention/invalid categories require a new figure design.")
        values = [votes.get(key, 0) for key in CATEGORIES]
        if sum(values) != 68:
            raise ValueError("Every attempt must remain in the denominator.")
        if abs(group["source_label_agreement_pct"]
               - 100 * group["source_label_matches"] / 68) > 1e-8:
            raise ValueError("Source-label summary does not agree with its count.")
        counts.append(values)
    return np.asarray(counts, dtype=int)


def draw(counts, output):
    for path in Path("/usr/share/fonts/urw-base35").glob("NimbusRoman-*.otf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 0,
        "ytick.major.size": 2.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(3.34, 2.0), dpi=200)
    fig.subplots_adjust(left=0.155, right=0.99, bottom=0.18, top=0.82)
    positions = np.arange(3)
    bottom = np.zeros(3, dtype=int)
    for category, color in enumerate(COLORS):
        values = counts[:, category]
        ax.bar(positions, values, 0.52, bottom=bottom, color=color,
               edgecolor="#222222", linewidth=0.5, zorder=3)
        for i, number in enumerate(values):
            if number:
                ax.text(positions[i], bottom[i] + number / 2, str(number),
                        ha="center", va="center", color="white",
                        fontsize=8.8, zorder=5)
        bottom += values
    ax.set_xticks(positions, LABELS)
    ax.set_ylabel("Episodes", labelpad=5)
    ax.set_ylim(0, 68)
    ax.set_yticks([0, 20, 40, 60, 68])
    ax.set_xlim(-0.55, 2.55)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.45, 2.45)
    ax.tick_params(axis="x", pad=5)
    ax.tick_params(axis="y", pad=3)
    handles = [Patch(facecolor=color, edgecolor="#222222", linewidth=0.5)
               for color in COLORS]
    fig.legend(handles, CATEGORIES, loc="upper center",
               bbox_to_anchor=(0.56, 1.0), ncol=2, frameon=False,
               handlelength=1.0, handleheight=0.9, handletextpad=0.5,
               columnspacing=1.6)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "photo_content_probe.pdf",
                metadata={"Title": CAPTION,
                          "Subject": "Selected-history final-response distributions; not native benchmark accuracy.",
                          "Creator": "Matplotlib"})
    fig.savefig(output / "photo_content_probe.png", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    counts = checked_counts(report)
    changes = {}
    for condition in ("gray", "shuffled"):
        transitions = report["paired_comparisons"]["real_minus_" + condition]["answer_transitions"]
        equal = sum(value for name, value in transitions.items()
                    if len(name.split(" -> ")) == 2
                    and name.split(" -> ")[0] == name.split(" -> ")[1])
        if sum(transitions.values()) != 68:
            raise ValueError("Paired transitions must include all 68 histories.")
        changes[condition] = {"changed_answers": 68 - equal,
                              "changed_answer_pct": 100 * (68 - equal) / 68,
                              "answer_transitions": transitions}
    data = {
        "report": str(args.report.resolve()),
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "scope": report["scope"], "caption": CAPTION,
        "histories_per_condition": 68, "conditions": list(CONDITIONS),
        "categories": list(CATEGORIES), "counts": counts.tolist(),
        "zero_categories_omitted": ["Need more evidence", "Invalid finish"],
        "paired_answer_changes": changes,
        "source_label_matches_for_body_only": {
            condition: report["conditions"][condition]["source_label_matches"]
            for condition in CONDITIONS},
        "source_label_agreement_pct_for_body_only": {
            condition: report["conditions"][condition]["source_label_agreement_pct"]
            for condition in CONDITIONS},
        "body_note": (
            "All conditions replay the same selected public history with one finish-only call. "
            "Only the newly returned photograph is replaced. Gray and shuffled versions both "
            "produce 68 Refuted responses, with no abstention or invalid output. These artificial "
            "images have no independently adjudicated expected answers. The figure shows final "
            "response distributions; it does not report native benchmark accuracy."
        ),
        "figure_size_inches": [3.34, 2.0],
        "font": "Nimbus Roman", "colors": list(COLORS),
    }
    draw(counts, args.output)
    (args.output / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "caption.tex").write_text(
        "\\caption{\\textbf{" + CAPTION + "}}\n"
        "\\label{fig:photo-content}\n")
    print(json.dumps({"complete": True, "output": str(args.output),
                      "counts": counts.tolist(), "paired_answer_changes": changes}))


if __name__ == "__main__":
    main()

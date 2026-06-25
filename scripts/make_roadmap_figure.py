#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


MAIN_STEPS = [
    {
        "id": "0",
        "title": "Done: Baselines",
        "body": "CIFAR-10 classifier: 95.29%\nDeepJSCC CBR: 1/3\nquality + semantic curves ready",
        "color": "#dceefb",
        "edge": "#2f80c0",
        "xy": (0.18, 0.70),
    },
    {
        "id": "1",
        "title": "Now: Fragility Feasibility",
        "body": "Run EXP-S2-002\nfragility vs random / saliency / gradient\nheld-out deletion, AUC, CI",
        "color": "#fff3cd",
        "edge": "#d99a00",
        "xy": (0.50, 0.70),
    },
    {
        "id": "2",
        "title": "Gate A",
        "body": "Pass only if fragility predicts\nheld-out semantic failure\nbetter than strongest baseline",
        "color": "#ffe2e2",
        "edge": "#c24141",
        "xy": (0.82, 0.70),
    },
    {
        "id": "3",
        "title": "Fixed-Budget Protection",
        "body": "Map scores to unequal protection\nsame CBR, total power, symbols\nside information counted",
        "color": "#e4f8e7",
        "edge": "#2f9e44",
        "xy": (0.82, 0.43),
    },
    {
        "id": "4",
        "title": "Gate B",
        "body": "Claim gain only if same-budget\nallocation improves accuracy,\nconsistency, failure rate",
        "color": "#ffe2e2",
        "edge": "#c24141",
        "xy": (0.50, 0.43),
    },
    {
        "id": "5",
        "title": "Deployable Predictor",
        "body": "Learn P_frag(F, SNR/CSI) -> r\navoid interventions at inference\nmeasure runtime + overhead",
        "color": "#eee7ff",
        "edge": "#7b61d1",
        "xy": (0.18, 0.43),
    },
    {
        "id": "6",
        "title": "Full Experiments",
        "body": "More datasets and channels\nSNR / CBR sweeps, ablations\ncomplexity + significance",
        "color": "#edf2ff",
        "edge": "#4263eb",
        "xy": (0.18, 0.19),
    },
    {
        "id": "7",
        "title": "Paper Package",
        "body": "Final claim, figures, tables\nreproducibility checklist\nsubmission material",
        "color": "#f8f0ff",
        "edge": "#9c36b5",
        "xy": (0.50, 0.19),
    },
]


REFINEMENT_STEPS = [
    {
        "title": "If Gate A fails",
        "body": "Refine intervention model,\nsemantic distance, feature grouping,\nand Monte Carlo variance",
        "x": 0.66,
        "target": 1,
    },
    {
        "title": "If Gate B fails",
        "body": "Revise allocation rule,\nside-information accounting,\nand fairness constraints",
        "x": 0.66,
        "target": 3,
    },
]


def add_box(ax, center_x, center_y, width, height, title, body, color, edge):
    left = center_x - width / 2.0
    bottom = center_y - height / 2.0
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color,
        edgecolor=edge,
        linewidth=2.0,
    )
    ax.add_patch(patch)
    ax.text(
        center_x,
        center_y + height * 0.23,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1f2933",
    )
    ax.text(
        center_x,
        center_y - height * 0.10,
        body,
        ha="center",
        va="center",
        fontsize=8.2,
        color="#1f2933",
        linespacing=1.25,
    )


def add_arrow(ax, start, end, color="#667085", rad=0.0, label=None, label_y_offset=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.9,
        color=color,
    )
    ax.add_patch(arrow)
    if label:
        lx = (start[0] + end[0]) / 2.0
        ly = (start[1] + end[1]) / 2.0 + label_y_offset
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            color=color,
            fontweight="bold",
        )


def draw_roadmap(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Future Roadmap: Channel-Conditioned Semantic Fragility-Aware DeepJSCC",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.895,
        "Baseline evidence -> fragility ranking gate -> same-budget protection gate -> deployable predictor -> full experiments -> paper",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )

    width = 0.25
    height = 0.18

    for step in MAIN_STEPS:
        x, y = step["xy"]
        add_box(
            ax,
            center_x=x,
            center_y=y,
            width=width,
            height=height,
            title=f"{step['id']}. {step['title']}",
            body=step["body"],
            color=step["color"],
            edge=step["edge"],
        )

    xy = [step["xy"] for step in MAIN_STEPS]
    connectors = [
        (0, 1, "next"),
        (1, 2, "test"),
        (2, 3, "pass"),
        (3, 4, "evaluate"),
        (4, 5, "pass"),
        (5, 6, "scale"),
        (6, 7, "write"),
    ]
    for start_index, end_index, label in connectors:
        start = xy[start_index]
        end = xy[end_index]
        if start[1] == end[1]:
            direction = 1.0 if end[0] > start[0] else -1.0
            start_point = (start[0] + direction * (width / 2.0 + 0.012), start[1])
            end_point = (end[0] - direction * (width / 2.0 + 0.012), end[1])
            label_offset = 0.035
            rad = 0.0
        else:
            start_point = (start[0], start[1] - height / 2.0 - 0.01)
            end_point = (end[0], end[1] + height / 2.0 + 0.01)
            label_offset = 0.0
            rad = 0.0
        add_arrow(
            ax,
            start_point,
            end_point,
            label=label,
            label_y_offset=label_offset,
            rad=rad,
        )

    ax.text(
        MAIN_STEPS[1]["xy"][0],
        MAIN_STEPS[1]["xy"][1] + height / 2.0 + 0.052,
        "Current highest-priority task",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#b7791f",
    )
    add_arrow(
        ax,
        (MAIN_STEPS[1]["xy"][0], MAIN_STEPS[1]["xy"][1] + height / 2.0 + 0.040),
        (MAIN_STEPS[1]["xy"][0], MAIN_STEPS[1]["xy"][1] + height / 2.0 + 0.005),
        color="#b7791f",
    )

    ax.text(
        0.82,
        0.20,
        "Fail-safe rule\nIf a gate fails, do not claim success.\nIterate on the specific cause, then rerun\nwith a new experiment ID.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5),
        linespacing=1.25,
    )

    ax.text(
        0.5,
        0.045,
        "Reporting stance: Stage 1 is evidence for the baseline; Stage 2 and Stage 3 are the two decision points that determine whether the idea becomes a paper.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f9fafb", edgecolor="#d1d5db"),
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_talk_track(output_path: Path) -> None:
    lines = [
        "# Future Roadmap Talk Track",
        "",
        "This roadmap is a presentation aid, not a completed-result claim.",
        "",
        "## One-sentence story",
        "",
        "We first establish a reproducible DeepJSCC baseline, then test whether channel-conditioned semantic fragility is a better predictor of semantic failure, then use that predictor to protect the most fragile transmitted features under exactly the same communication budget.",
        "",
        "## Step-by-step",
        "",
        "1. Baseline already done: CIFAR-10 classifier and AWGN DeepJSCC are trained and evaluated, so we know how image quality and semantic accuracy change with SNR.",
        "2. Next decision point: run `EXP-S2-002` to test whether semantic fragility ranks risky feature groups better than random, activation saliency, and channel-aware gradient baselines.",
        "3. Gate A: if fragility does not win with held-out corruption, deletion AUC, and confidence intervals, revise the intervention definition, semantic distance, feature grouping, or Monte Carlo setting before moving on.",
        "4. Fixed-budget protection: if Gate A passes, convert fragility scores into unequal protection or power allocation while keeping CBR, total power, symbols, and side-information cost fair.",
        "5. Gate B: only claim a communication advantage if the same-budget allocation improves semantic robustness across SNRs.",
        "6. Deployable predictor: replace expensive oracle interventions with a lightweight predictor `P_frag(F, SNR/CSI) -> r` and measure runtime, parameters, and side-information overhead.",
        "7. Full experimental story: expand to more datasets, channels, SNR/CBR sweeps, ablations, complexity analysis, and significance tests before writing the paper.",
        "",
        "## Generated files",
        "",
        "- `future_roadmap.png`: slide-ready roadmap.",
        "- `future_roadmap.md`: this talk track.",
        "- `future_roadmap.mmd`: Mermaid source for quick editing in slide tools.",
        "- `asset_manifest.json`: file index.",
        "",
        "## Caveat",
        "",
        "Do not present Stage 2 or Stage 3 as completed. The current verified result is still the Stage 1 baseline; the roadmap shows the planned route to a defensible paper claim.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mermaid(output_path: Path) -> None:
    mermaid = """flowchart LR
    A[Done: Baselines<br/>Classifier + DeepJSCC<br/>EXP-S1-004 / EXP-S1-005]
    B[Now: Fragility Feasibility<br/>Run EXP-S2-002<br/>Fragility vs baselines]
    C{Gate A<br/>Better held-out ranking?}
    D[Fixed-Budget Allocation<br/>Same CBR / power / symbols<br/>side information counted]
    E{Gate B<br/>Same-budget semantic gain?}
    F[Deployable Predictor<br/>P_frag(F, SNR/CSI) -> r<br/>low overhead]
    G[Full Experimental Story<br/>datasets / channels / ablations<br/>complexity / paper]
    H[Refine intervention,<br/>semantic distance,<br/>grouping, MC variance]
    I[Revise allocation rule<br/>and side-info accounting]

    A --> B --> C
    C -- yes --> D --> E
    C -- no --> H --> B
    E -- yes --> F --> G
    E -- no --> I --> D
"""
    output_path.write_text(mermaid, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/report_assets/future_roadmap",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    png_path = args.output_dir / "future_roadmap.png"
    draw_roadmap(png_path)
    write_talk_track(args.output_dir / "future_roadmap.md")
    write_mermaid(args.output_dir / "future_roadmap.mmd")

    manifest = {
        "purpose": "future research roadmap for presentation",
        "source_documents": ["PROJECT.md", "PROGRESS.md", "EXPERIMENTS.md"],
        "outputs": sorted(path.name for path in args.output_dir.iterdir()),
    }
    (args.output_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

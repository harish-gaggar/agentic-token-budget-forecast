"""Regenerate the generic-agent persona figure from results/generic_agents*.csv.

Reads two CSVs produced by experiments/generic_agents.py and writes a
two-panel figure: (left) before/after tokens per persona on a log
y-axis, (right) per-persona stacked bar of tokens saved attributed to
each layer of the funnel. The figure appears in the "Generic-agent
personas" subsection of paper.tex.

Usage::

    python -m experiments.plot_generic_agents --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# Stable persona ordering for the paper.
PERSONAS = ["react_web_search", "code_fix", "multi_turn_research", "polling_agent"]
PERSONA_LABELS = {
    "react_web_search":    "ReAct\nweb search",
    "code_fix":            "Code-fix\n(file re-reads)",
    "multi_turn_research": "Multi-turn\nresearch",
    "polling_agent":       "Polling\n(retry-style)",
}

# Stable layer ordering and short labels for the stacked bar.
LAYER_ORDER = [
    "compression:dedup_messages",
    "compression:tool_results",
    "clustering:repeated_tool_calls",
    "reranking:composite",
    "selection:token_budget",
]
LAYER_LABELS = {
    "compression:dedup_messages":      "C: dedup msgs",
    "compression:tool_results":        "C: tool results",
    "clustering:repeated_tool_calls":  "Cl: repeated calls",
    "reranking:composite":             "R: composite",
    "selection:token_budget":          "S: token budget",
}
LAYER_COLOURS = {
    "compression:dedup_messages":      "#a0aec0",
    "compression:tool_results":        "#1a365d",
    "clustering:repeated_tool_calls":  "#2b6cb0",
    "reranking:composite":             "#9ae6b4",
    "selection:token_budget":          "#d69e2e",
}


def _plot(summary: pd.DataFrame, layers: pd.DataFrame, path: Path) -> None:
    # Aggregate per persona (mean across seeds)
    agg = (summary.groupby("persona", as_index=False)
                  .agg(tokens_before=("tokens_before", "mean"),
                       tokens_after=("tokens_after", "mean"),
                       compression_ratio=("compression_ratio", "mean")))
    agg["persona_order"] = agg["persona"].map({p: i for i, p in enumerate(PERSONAS)})
    agg = agg.sort_values("persona_order").reset_index(drop=True)

    layer_agg = (layers.groupby(["persona", "layer"], as_index=False)
                       .agg(mean_saved=("saved_tokens", "mean")))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # --- Left: per-persona before/after (log scale) ---
    x = range(len(agg))
    width = 0.38
    before = agg["tokens_before"].tolist()
    after = agg["tokens_after"].tolist()
    bars_before = axL.bar(
        [i - width / 2 for i in x], before, width,
        label="Without optimizer",
        color="#bbb", edgecolor="black", linewidth=0.4,
    )
    bars_after = axL.bar(
        [i + width / 2 for i in x], after, width,
        label="With optimizer (balanced)",
        color="#1a365d", edgecolor="black", linewidth=0.4,
    )
    for bar, v in list(zip(bars_before, before)) + list(zip(bars_after, after)):
        axL.text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                 f"{int(v):,}", ha="center", va="bottom", fontsize=7)

    axL.set_yscale("log")
    axL.set_ylim(100, max(before) * 2.5)
    axL.set_ylabel("Mean input tokens / run (log scale)")
    axL.set_xticks(list(x))
    axL.set_xticklabels([PERSONA_LABELS[p] for p in agg["persona"]], fontsize=8)
    ratios = " | ".join(
        f"{p.split('_')[0]}: {r:.0%}"
        for p, r in zip(agg["persona"], agg["compression_ratio"])
    )
    axL.set_title("Before vs. after, per persona (25 seeds each)\n"
                  f"compression ratio = {ratios}",
                  fontsize=9)
    axL.legend(fontsize=7, loc="upper right", framealpha=0.95)
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)

    # --- Right: stacked bar of saved tokens by layer, per persona ---
    bottoms = [0.0] * len(agg)
    bar_x = list(range(len(agg)))
    for layer in LAYER_ORDER:
        heights = []
        for persona in agg["persona"]:
            row = layer_agg[(layer_agg["persona"] == persona)
                            & (layer_agg["layer"] == layer)]
            heights.append(float(row["mean_saved"].iloc[0]) if not row.empty else 0.0)
        axR.bar(bar_x, heights, bottom=bottoms,
                color=LAYER_COLOURS[layer], edgecolor="black", linewidth=0.4,
                label=LAYER_LABELS[layer])
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    for i, total in enumerate(bottoms):
        if total > 0:
            axR.text(i, total * 1.02, f"{int(total):,}",
                     ha="center", va="bottom", fontsize=8)

    axR.set_xticks(bar_x)
    axR.set_xticklabels([PERSONA_LABELS[p] for p in agg["persona"]], fontsize=8)
    axR.set_ylabel("Mean tokens saved / run (by layer)")
    axR.set_title("Layer attribution per persona", fontsize=9)
    axR.legend(fontsize=7, loc="upper right", framealpha=0.95, ncol=1)
    axR.spines["top"].set_visible(False)
    axR.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    summary = pd.read_csv(out_dir / "generic_agents.csv")
    layers = pd.read_csv(out_dir / "generic_agents_layers.csv")
    _plot(summary, layers, out_dir / "fig_generic_agents.pdf")
    print(f"Wrote fig_generic_agents.pdf to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

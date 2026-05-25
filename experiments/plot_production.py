"""Regenerate the production case-study figures from CSVs in results/.

This script reads CSVs in ``results/`` and writes PDF figures next to
them. The CSVs come from a live deployment of ContextOptimizer inside
a LangGraph BigQuery NL-to-SQL agent. The figures appear in Section 8
of ``paper.tex``.

Usage::

    python -m experiments.plot_production --out results

Inputs:
    results/production_token_savings_samples.csv
    results/production_token_savings_summary.csv
    results/production_cost_projections.csv
    results/optimizer_microbench_layers.csv  (optional)

Outputs:
    results/fig_production_savings.pdf
    results/fig_production_costs.pdf
    results/fig_layer_attribution.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(out_dir / "production_token_savings_samples.csv")
    # `value` is mixed (ints, floats, strings); coerce numeric where possible
    # so callers can safely apply int()/float() without surprises.
    summary_df = pd.read_csv(out_dir / "production_token_savings_summary.csv")

    def _coerce(v: object) -> object:
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    summary_df["value"] = summary_df["value"].map(_coerce)
    summary = summary_df.set_index("metric")["value"]
    costs = pd.read_csv(out_dir / "production_cost_projections.csv")
    return samples, summary, costs


def _plot_savings(samples: pd.DataFrame, summary: pd.Series, path: Path) -> None:
    """Grouped bar chart: per-sample before/after tokens + aggregate column."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    # Per-sample (real log lines) on the left.
    labels = [f"call {i+1}" for i in range(len(samples))] + ["aggregate (14 calls)"]
    before = list(samples["before_tokens"]) + [float(summary["tokens_without_optimizer"])]
    after = list(samples["after_tokens"]) + [float(summary["tokens_with_optimizer"])]

    x = range(len(labels))
    width = 0.38
    bars_before = ax.bar(
        [i - width / 2 for i in x], before, width,
        label="Without optimizer",
        color="#bbb", edgecolor="black", linewidth=0.4,
    )
    bars_after = ax.bar(
        [i + width / 2 for i in x], after, width,
        label="With optimizer (balanced preset)",
        color="#1a365d", edgecolor="black", linewidth=0.4,
    )
    for bar, v in list(zip(bars_before, before)) + list(zip(bars_after, after)):
        # Skip annotations on the aggregate column to avoid crowding.
        if v > 30_000:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(before) * 0.012,
                f"{int(v):,}", ha="center", va="bottom", fontsize=7)

    ax.set_yscale("log")
    ax.set_ylabel("Input tokens (log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(
        "Per-call input tokens before vs. after ContextOptimizer "
        f"({float(summary['reduction_pct_overall']):.1f}% aggregate reduction)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_costs(costs: pd.DataFrame, path: Path) -> None:
    """Annualised cost-saved bars across three model price points.

    The numbers are read straight from the CSV. The CSV documents
    the derivation: avg input tokens per turn (from 75 production
    turns) times the measured 72% reduction, applied to public list
    prices captured at the paper's submission date.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = costs["model"].tolist()
    vals = costs["annual_1k_per_day_usd"].astype(float).tolist()
    colours = ["#9ae6b4", "#2b6cb0", "#1a365d"]
    bars = ax.bar(labels, vals, color=colours, edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Annualised savings (USD)")
    ax.set_title(
        "Projected annual savings at 1{,}000 turns/day\n"
        "(avg input/turn * measured 72% reduction; input tokens only)",
        fontsize=10,
    )
    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_layer_attribution(out_dir: Path, path: Path) -> None:
    """Per-layer token attribution + per-layer wall-clock, from real microbench."""
    layers_csv = out_dir / "optimizer_microbench_layers.csv"
    if not layers_csv.exists():
        print(f"  skip layer attribution: {layers_csv} not present "
              "(run optimizer_microbench.py first)")
        return
    df = pd.read_csv(layers_csv)
    agg = (df.groupby("layer", as_index=False)
             .agg(mean_saved_tokens=("saved_tokens", "mean"),
                  mean_ms=("elapsed_ms", "mean"),
                  n=("layer", "count")))
    # Stable display order matching the funnel.
    order = [
        "compression:dedup_messages",
        "compression:tool_results",
        "clustering:repeated_tool_calls",
        "reranking:composite",
        "selection:token_budget",
    ]
    agg["order"] = agg["layer"].map({l: i for i, l in enumerate(order)})
    agg = agg.sort_values("order").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))
    labels = [L.replace("compression:", "C:")
              .replace("clustering:", "Cl:")
              .replace("reranking:", "R:")
              .replace("selection:", "S:")
              for L in agg["layer"]]
    saved = agg["mean_saved_tokens"].tolist()
    ms = agg["mean_ms"].tolist()
    colours = ["#1a365d" if v > 0 else "#bbb" for v in saved]

    bars1 = ax1.bar(labels, saved, color=colours, edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars1, saved):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 v + max(saved) * 0.02 if max(saved) > 0 else 0,
                 f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Mean saved tokens / run")
    ax1.set_title("Token attribution (100 runs, balanced preset)", fontsize=10)
    ax1.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bars2 = ax2.bar(labels, ms, color="#2b6cb0", edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars2, ms):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + max(ms) * 0.02,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Mean wall-clock per run (ms)")
    ax2.set_title("Per-layer overhead", fontsize=10)
    ax2.tick_params(axis="x", labelrotation=25, labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    samples, summary, costs = _load(out_dir)
    _plot_savings(samples, summary, out_dir / "fig_production_savings.pdf")
    _plot_costs(costs, out_dir / "fig_production_costs.pdf")
    _plot_layer_attribution(out_dir, out_dir / "fig_layer_attribution.pdf")
    print(f"Wrote fig_production_savings.pdf, fig_production_costs.pdf, "
          f"fig_layer_attribution.pdf to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

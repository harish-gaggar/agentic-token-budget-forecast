"""Stagewise prediction quality for TABF and the strongest baselines.

The reviewer pointed out:
  "The GBT forecaster predicts total tokens and then redistributes
   per-stage using heuristic ratios; stagewise accuracy is not
   quantitatively evaluated, limiting claims about per-stage fidelity."

We address that directly. For every prediction on the held-out test
set we compare the per-stage forecast against the simulator's per-stage
ground truth. MAE, MAPE, and W20R are reported per stage so the
reader can see exactly where the redistribution-by-heuristic-ratio
strategy works and where it breaks down.

Output:
  results/stagewise_results.csv   (method, stage, mae, mape, w20r, n)
  results/fig_stagewise.pdf       grouped-bar chart
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabf.extractor import extract_signals
from tabf.classifier import classify, TASK_TYPES
from tabf.heuristic import HeuristicBudgetModel
from tabf.ml import GBTBudgetModel

from benchmark.corpus import build_corpus
from benchmark.simulator import simulate_dataset
from experiments.metrics import mae, mape, w20r

CORPUS_SEED = 1729
SIM_SEED    = 20260514
N_TASKS     = 600
TRAIN_FRAC  = 0.7

STAGES = ["system", "tools", "memory", "context", "reasoning", "output"]
STAGE_LABEL = {
    "system":    "System",
    "tools":     "Tools",
    "memory":    "Memory",
    "context":   "Retrieved\ncontext",
    "reasoning": "Reasoning",
    "output":    "Output",
}


def _stage_pred(model, task: str) -> Dict[str, int]:
    sig = extract_signals(task)
    ktype, cx = classify(task, sig)
    fc = model.forecast(ktype, cx, sig)
    return {
        "system":    fc.system,
        "tools":     fc.tools,
        "memory":    fc.memory,
        "context":   fc.context,
        "reasoning": fc.reasoning,
        "output":    fc.output,
    }


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(N_TASKS, seed=CORPUS_SEED)
    rows = simulate_dataset(corpus, seed=SIM_SEED)
    rng = np.random.default_rng(42)
    by_type: Dict[str, list] = {t: [] for t in TASK_TYPES}
    for r in rows:
        by_type[r["latent_type"]].append(r)
    train, test = [], []
    for t, lst in by_type.items():
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        split = int(len(lst) * TRAIN_FRAC)
        train.extend(lst[i] for i in idx[:split])
        test.extend(lst[i] for i in idx[split:])

    heur = HeuristicBudgetModel()
    train_traces = []
    for r in train:
        sig = extract_signals(r["text"])
        ktype, cx = classify(r["text"], sig)
        train_traces.append((sig, ktype, cx, r["gt"]["input_tokens"], r["gt"]["output"]))
    gbt = GBTBudgetModel(heuristic=heur, random_state=0).fit(train_traces)

    methods = [("TABF_heuristic", heur), ("TABF_gbt", gbt)]
    rows_out: List[Dict] = []
    for mname, model in methods:
        # Per-stage prediction and truth vectors over the test set.
        preds_by_stage = {s: [] for s in STAGES}
        truth_by_stage = {s: [] for s in STAGES}
        for r in test:
            sp = _stage_pred(model, r["text"])
            for s in STAGES:
                preds_by_stage[s].append(sp[s])
                truth_by_stage[s].append(r["gt"][s])

        for s in STAGES:
            p = np.asarray(preds_by_stage[s], dtype=float)
            y = np.asarray(truth_by_stage[s], dtype=float)
            rows_out.append({
                "method": mname,
                "stage":  s,
                "n":      int(len(p)),
                "mae":    round(float(mae(p, y)), 1),
                "mape":   round(float(mape(p, y)), 2),
                "w20r":   round(float(w20r(p, y)), 2),
                "mean_truth": int(round(float(np.mean(y)))),
                "mean_pred":  int(round(float(np.mean(p)))),
            })

    df = pd.DataFrame(rows_out)
    df.to_csv(out_dir / "stagewise_results.csv", index=False)

    # Console report.
    print("Per-stage error (test set, n=185):")
    print(df.to_string(index=False))

    # Plot: grouped bars of W20R per stage, one bar per method.
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(STAGES))
    width = 0.4
    heur_w20r = [df[(df["method"] == "TABF_heuristic") & (df["stage"] == s)]
                 ["w20r"].iloc[0] for s in STAGES]
    gbt_w20r = [df[(df["method"] == "TABF_gbt") & (df["stage"] == s)]
                ["w20r"].iloc[0] for s in STAGES]
    ax.bar(x - width/2, heur_w20r, width, label="TABF Heuristic",
           color="#2b6cb0", edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, gbt_w20r,  width, label="TABF + GBT",
           color="#1a365d", edgecolor="black", linewidth=0.4)
    for i, (h, g) in enumerate(zip(heur_w20r, gbt_w20r)):
        ax.text(i - width/2, h + 1.0, f"{h:.0f}", ha="center", fontsize=8)
        ax.text(i + width/2, g + 1.0, f"{g:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
    ax.set_ylabel("Within-20% rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_stagewise.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out_dir/'stagewise_results.csv'} and "
          f"{out_dir/'fig_stagewise.pdf'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

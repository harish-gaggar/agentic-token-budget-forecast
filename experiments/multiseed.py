"""Multi-seed cross-validation of TABF and the extended baselines.

The reviewer asked for "multiple random train/test splits or
cross-validation to assess variability." This script re-runs the full
eval pipeline of experiments.run_all with K different (corpus_seed,
sim_seed, split_seed) tuples and reports mean +/- std for every
method on MAE, MAPE, and W20R.

We hold the *corpus generator* fixed across seeds (same 600 task
descriptions) and vary the simulator seed + the train/test split.
This isolates two sources of variance:

  * simulator noise: the ground-truth simulator is stochastic; the
    same task description gets a different trace each seed.
  * split noise: which 415 of the 600 tasks land in train vs the
    held-out 185 changes the test distribution slightly.

Both are realistic noise sources for "how much should I trust the
single-split numbers in Table~main?"

Output:
  results/multiseed_results.csv   one row per (method, metric, seed)
  results/multiseed_summary.csv   mean +/- std per (method, metric)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from tabf.extractor import extract_signals
from tabf.classifier import classify, TASK_TYPES
from tabf.heuristic import HeuristicBudgetModel
from tabf.ml import GBTBudgetModel

from benchmark.corpus import build_corpus
from benchmark.simulator import simulate_dataset

from experiments.baselines import (
    UniformBaseline, TypeOnlyHeuristic, LengthProportional,
    RidgeOnFeatures, RandomForestOnFeatures, HistGBOnFeatures,
    RidgeOnTfidf, HistGBOnTfidf,
)
from experiments.metrics import mae, mape, w20r
from experiments.run_all import _evaluate, CORPUS_SEED, N_TASKS, TRAIN_FRAC


def _one_seed(sim_seed: int, split_seed: int):
    corpus = build_corpus(N_TASKS, seed=CORPUS_SEED)
    rows = simulate_dataset(corpus, seed=sim_seed)

    rng = np.random.default_rng(split_seed)
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

    train_texts = [r["text"] for r in train]
    train_totals = [r["gt"]["total"] for r in train]

    uniform = UniformBaseline().fit(train_totals)
    length = LengthProportional().fit(
        [len(r["text"].split()) for r in train], train_totals,
    )
    type_only = TypeOnlyHeuristic()
    heur = HeuristicBudgetModel()
    ridge_h  = RidgeOnFeatures().fit(train_texts, train_totals)
    rf_h     = RandomForestOnFeatures(random_state=split_seed).fit(train_texts, train_totals)
    histgb_h = HistGBOnFeatures(random_state=split_seed).fit(train_texts, train_totals)
    ridge_t  = RidgeOnTfidf().fit(train_texts, train_totals)
    histgb_t = HistGBOnTfidf().fit(train_texts, train_totals)

    train_traces = []
    for r in train:
        sig = extract_signals(r["text"])
        ktype, cx = classify(r["text"], sig)
        train_traces.append((sig, ktype, cx, r["gt"]["input_tokens"], r["gt"]["output"]))
    gbt = GBTBudgetModel(heuristic=heur, random_state=split_seed).fit(train_traces)

    models = [
        ("B1_uniform",      uniform),
        ("B2_type_only",    type_only),
        ("B3_length",       length),
        ("B4_ridge_hand",   ridge_h),
        ("B5_rf_hand",      rf_h),
        ("B6_histgb_hand",  histgb_h),
        ("B7_ridge_tfidf",  ridge_t),
        ("B8_histgb_tfidf", histgb_t),
        ("TABF_heuristic",  heur),
        ("TABF_gbt",        gbt),
    ]

    out = {}
    for name, model in models:
        r = _evaluate(model, test)
        out[name] = {"mae": r["mae"], "mape": r["mape"], "w20r": r["w20r"]}
    return out


def main(out_dir: Path, n_seeds: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Five (sim, split) tuples. The sim seed varies the ground-truth
    # trace; the split seed varies the train/test partition. We use a
    # one-line LCG mix so the pair is deterministic from the index.
    seeds = [(20260514 + 1_000_003 * i, 42 + 97 * i) for i in range(n_seeds)]

    long_rows = []
    for seed_idx, (sim_seed, split_seed) in enumerate(seeds):
        print(f"[seed {seed_idx+1}/{n_seeds}] sim={sim_seed}  split={split_seed}")
        res = _one_seed(sim_seed, split_seed)
        for method, m in res.items():
            for metric, value in m.items():
                long_rows.append({
                    "seed_idx": seed_idx,
                    "sim_seed": sim_seed,
                    "split_seed": split_seed,
                    "method": method,
                    "metric": metric,
                    "value": float(value),
                })
            print(f"  {method:18s}  MAE={m['mae']:7.1f}  "
                  f"MAPE={m['mape']:5.2f}%  W20R={m['w20r']:5.2f}%")

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / "multiseed_results.csv", index=False)

    # Summary: mean and std per (method, metric).
    summary_rows = []
    for (method, metric), grp in long_df.groupby(["method", "metric"]):
        summary_rows.append({
            "method": method,
            "metric": metric,
            "n_seeds": int(len(grp)),
            "mean": float(grp["value"].mean()),
            "std": float(grp["value"].std(ddof=1)),
            "min": float(grp["value"].min()),
            "max": float(grp["value"].max()),
        })
    sum_df = pd.DataFrame(summary_rows).sort_values(["method", "metric"])
    sum_df.to_csv(out_dir / "multiseed_summary.csv", index=False)

    print("\nSummary (mean +/- std across seeds):")
    for method in sum_df["method"].unique():
        row_mae  = sum_df[(sum_df["method"] == method) & (sum_df["metric"] == "mae")].iloc[0]
        row_mape = sum_df[(sum_df["method"] == method) & (sum_df["metric"] == "mape")].iloc[0]
        row_w20  = sum_df[(sum_df["method"] == method) & (sum_df["metric"] == "w20r")].iloc[0]
        print(f"  {method:18s}  "
              f"MAE={row_mae['mean']:7.1f}+/-{row_mae['std']:5.1f}  "
              f"MAPE={row_mape['mean']:5.2f}+/-{row_mape['std']:4.2f}%  "
              f"W20R={row_w20['mean']:5.2f}+/-{row_w20['std']:4.2f}%")

    print("\nWrote results to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--seeds", default=5, type=int)
    args = ap.parse_args()
    main(args.out, args.seeds)

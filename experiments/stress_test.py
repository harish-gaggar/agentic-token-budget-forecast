"""Simulator stress test for TABF.

The reviewer pointed out the circularity risk:
  "The synthetic simulator bakes in signal-dependent effects that
   the heuristic only partially models, which may advantage the
   proposed features... external validity remains the key risk."

We address this directly by re-running TABF against two perturbed
simulators that deliberately *break* assumptions the heuristic was
calibrated on, then comparing degradation in MAE, MAPE, and W20R
between the canonical simulator and each perturbation. If TABF
collapses, that means the heuristic was overfitting to the simulator;
if TABF degrades gracefully, it means the predictor is picking up on
features that survive moderate distribution shift.

Perturbations:
  v1_heavy_reasoning: reasoning stage means are scaled by 2x on every
    task type. Models the worst case of "the agent does more
    intermediate reasoning than we baked into the heuristic table."
  v2_shuffled_ratios: stage-mean vectors are randomly permuted *across
    task types* (deterministic per seed). Models the worst case of
    "the per-type token signature is completely different in your
    deployment than in ours" (e.g., a SQL agent vs a code-fix agent
    use the same stages but in inverted proportions).
  v3_amplified_noise: per-stage log-normal sigmas are doubled. Models
    "your traces are noisier than ours."

Output:
  results/stress_test_results.csv     long-form with one row per
                                      (sim_variant, method, metric).
  results/stress_test_summary.csv     wide-form table for inclusion
                                      in the paper.
"""

from __future__ import annotations
import argparse
import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tabf.extractor import extract_signals
from tabf.classifier import classify, TASK_TYPES
from tabf.heuristic import HeuristicBudgetModel
from tabf.ml import GBTBudgetModel

from benchmark.corpus import build_corpus
from benchmark.simulator import (
    simulate_dataset, SIM_BASE, _DEFAULT_SIGMA, _OUTPUT_SIGMA_BY_TYPE,
)
from experiments.baselines import (
    RandomForestOnFeatures, HistGBOnFeatures,
)
from experiments.metrics import mae, mape, w20r
from experiments.run_all import _evaluate

CORPUS_SEED = 1729
SIM_SEED    = 20260514
N_TASKS     = 600
TRAIN_FRAC  = 0.7


def _split(rows: List[dict], split_seed: int):
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
    return train, test


def _simulate_variant(corpus: List[dict], variant: str, sim_seed: int) -> List[dict]:
    """Apply a perturbation to the simulator's globals, run, then restore."""
    orig_base = copy.deepcopy(SIM_BASE)
    orig_sigma = copy.deepcopy(_DEFAULT_SIGMA)
    orig_out_sigma = copy.deepcopy(_OUTPUT_SIGMA_BY_TYPE)

    try:
        if variant == "v0_canonical":
            pass  # unmodified
        elif variant == "v1_heavy_reasoning":
            for t in SIM_BASE:
                s, to, m, c, r, o = SIM_BASE[t]
                SIM_BASE[t] = (s, to, m, c, int(r * 2.0), o)
        elif variant == "v2_shuffled_ratios":
            rng = np.random.default_rng(20260530)
            keys = list(SIM_BASE.keys())
            vals = [SIM_BASE[k] for k in keys]
            order = list(range(len(vals)))
            rng.shuffle(order)
            for k, j in zip(keys, order):
                SIM_BASE[k] = vals[j]
        elif variant == "v3_amplified_noise":
            for k in list(_DEFAULT_SIGMA.keys()):
                _DEFAULT_SIGMA[k] = _DEFAULT_SIGMA[k] * 2.0
            for k in list(_OUTPUT_SIGMA_BY_TYPE.keys()):
                _OUTPUT_SIGMA_BY_TYPE[k] = _OUTPUT_SIGMA_BY_TYPE[k] * 2.0
        else:
            raise ValueError(f"unknown variant {variant}")

        rows = simulate_dataset(corpus, seed=sim_seed)
    finally:
        SIM_BASE.clear(); SIM_BASE.update(orig_base)
        _DEFAULT_SIGMA.clear(); _DEFAULT_SIGMA.update(orig_sigma)
        _OUTPUT_SIGMA_BY_TYPE.clear(); _OUTPUT_SIGMA_BY_TYPE.update(orig_out_sigma)

    return rows


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(N_TASKS, seed=CORPUS_SEED)

    variants = ["v0_canonical", "v1_heavy_reasoning",
                "v2_shuffled_ratios", "v3_amplified_noise"]

    long_rows: List[Dict] = []

    for variant in variants:
        print(f"\n=== variant: {variant} ===")
        rows = _simulate_variant(corpus, variant, sim_seed=SIM_SEED)
        train, test = _split(rows, split_seed=42)

        train_texts  = [r["text"] for r in train]
        train_totals = [r["gt"]["total"] for r in train]

        heur = HeuristicBudgetModel()
        rf_h    = RandomForestOnFeatures().fit(train_texts, train_totals)
        histgb_h = HistGBOnFeatures().fit(train_texts, train_totals)

        train_traces = []
        for r in train:
            sig = extract_signals(r["text"])
            ktype, cx = classify(r["text"], sig)
            train_traces.append(
                (sig, ktype, cx, r["gt"]["input_tokens"], r["gt"]["output"])
            )
        gbt = GBTBudgetModel(heuristic=heur, random_state=0).fit(train_traces)

        for name, model in [
            ("TABF_heuristic", heur),
            ("TABF_gbt",       gbt),
            ("B5_rf_hand",     rf_h),
            ("B6_histgb_hand", histgb_h),
        ]:
            r = _evaluate(model, test)
            for metric, value in [("mae", r["mae"]), ("mape", r["mape"]),
                                  ("w20r", r["w20r"])]:
                long_rows.append({
                    "variant": variant,
                    "method":  name,
                    "metric":  metric,
                    "value":   float(value),
                })
            print(f"  {name:18s}  MAE={r['mae']:7.1f}  "
                  f"MAPE={r['mape']:5.2f}%  W20R={r['w20r']:5.2f}%")

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / "stress_test_results.csv", index=False)

    # Wide-form summary table for the paper: rows = method, cols =
    # variant, one block per metric.
    wide = long_df.pivot_table(
        index=["method", "metric"], columns="variant", values="value"
    ).reset_index()
    wide.to_csv(out_dir / "stress_test_summary.csv", index=False)

    print("\nWide-form summary table:")
    print(wide.to_string(index=False))
    print(f"\nWrote {out_dir/'stress_test_results.csv'} and "
          f"{out_dir/'stress_test_summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

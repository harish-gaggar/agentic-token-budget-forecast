"""Real-trace validation of TABF on production turns + TABF-driven optimizer.

This is the experiment the reviewer asked for in two of their seven
questions:

  Q1 (real-trace validation):
    "TABF is not validated on real agent traces; all forecasting
     results come from a synthetic simulator..."

  Q7 (TABF-driven optimizer):
    "Have you experimented with integrating TABF's forecast to drive
     ContextOptimizer's max_tokens in production, and if so, what
     were the impacts on savings, latency, and error rates?"

We use the latest production extraction from
results/production_bq_turns.csv. Each row is one real production turn
of the deployed BigQuery LangGraph agent, with the actual task text
the user sent and the actual total tokens billed to OpenAI after
ContextOptimizer ran.

Stage 1. Pure forecasting accuracy on real traces.
  Run TABF (heuristic + GBT) on each task description and compare
  the predicted total against the actual observed total. Report
  MAE, MAPE, W20R, and the rank correlation with the actual order.
  This is the closest available analogue to "fit on synthetic,
  evaluate on real" - it tells us whether the predictions hold up
  outside the synthetic simulator.

Stage 2. TABF-driven ContextOptimizer simulation.
  For each turn, treat TABF's forecast as the proposed max_tokens.
  Count the turns where the actual observed usage falls within
  TABF's predicted budget (within budget), where it overshoots
  (would have been truncated under hard cap), and where it
  undershoots by a lot (we paid for less than the budget). Compare
  to a static-config baseline that just uses the corpus median.

Output:
  results/tabf_replay_forecast.csv   forecast vs actual per turn
  results/tabf_replay_summary.csv    rolled-up metrics
  results/tabf_replay_manifest.json  what we ran
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

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


def _train_models():
    """Train TABF heuristic + GBT on the synthetic corpus (no peeking)."""
    corpus = build_corpus(N_TASKS, seed=CORPUS_SEED)
    rows = simulate_dataset(corpus, seed=SIM_SEED)

    rng = np.random.default_rng(42)
    by_type: Dict[str, list] = {t: [] for t in TASK_TYPES}
    for r in rows:
        by_type[r["latent_type"]].append(r)
    train = []
    for t, lst in by_type.items():
        idx = np.arange(len(lst)); rng.shuffle(idx)
        split = int(len(lst) * TRAIN_FRAC)
        train.extend(lst[i] for i in idx[:split])

    heur = HeuristicBudgetModel()
    train_traces = []
    for r in train:
        sig = extract_signals(r["text"])
        ktype, cx = classify(r["text"], sig)
        train_traces.append((sig, ktype, cx, r["gt"]["input_tokens"], r["gt"]["output"]))
    gbt = GBTBudgetModel(heuristic=heur, random_state=0).fit(train_traces)

    # Also compute the corpus median total - used as the "static
    # config" baseline in stage 2.
    median_total = float(np.median([r["gt"]["total"] for r in train]))
    return heur, gbt, median_total


def _forecast_total(model, text: str) -> int:
    sig = extract_signals(text)
    ktype, cx = classify(text, sig)
    fc = model.forecast(ktype, cx, sig)
    return int(fc.total_tokens)


def main(out_dir: Path, prod_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    turns = pd.read_csv(prod_csv)
    # The 'input' column contains the original user task plus optional
    # SQL context. For TABF forecasting we use the *first paragraph*
    # only - that is the natural-language task the user submitted.
    # Everything after the first '---' or '###' delimiter is treated
    # as auxiliary context the agent will retrieve at runtime, not
    # part of the dispatch task.
    def _first_para(s: str) -> str:
        if not isinstance(s, str):
            return ""
        for sep in ("\n---", "\n###", "\n```"):
            j = s.find(sep)
            if j != -1:
                s = s[:j]
        return s.strip()
    turns["task_text"] = turns["input"].apply(_first_para)
    turns = turns[turns["task_text"].str.len() > 0].reset_index(drop=True)
    print(f"Production turns with usable task text: n={len(turns)}")

    heur, gbt, median_total = _train_models()
    print(f"Static-config baseline (train-corpus median total): {median_total:.0f}")

    # ---- Stage 1: forecast vs actual on real production turns. ----
    per_turn = []
    for _, r in turns.iterrows():
        pred_h = _forecast_total(heur, r["task_text"])
        pred_g = _forecast_total(gbt,  r["task_text"])
        actual = int(r["total_tokens"]) if pd.notna(r["total_tokens"]) else None
        if actual is None:
            continue
        per_turn.append({
            "request_id":       r["request_id"],
            "task_text":        r["task_text"][:140],
            "actual_total":     actual,
            "actual_input":     int(r["input_tokens"]),
            "actual_output":    int(r["output_tokens"]),
            "actual_llm_calls": int(r["llm_calls"]) if pd.notna(r["llm_calls"]) else None,
            "actual_tool_calls": int(r["tool_calls"]) if pd.notna(r["tool_calls"]) else None,
            "pred_heuristic":   pred_h,
            "pred_gbt":         pred_g,
            "static_baseline":  int(median_total),
        })
    pt_df = pd.DataFrame(per_turn)
    pt_df.to_csv(out_dir / "tabf_replay_forecast.csv", index=False)

    actual = pt_df["actual_total"].astype(float).values
    pred_h = pt_df["pred_heuristic"].astype(float).values
    pred_g = pt_df["pred_gbt"].astype(float).values
    pred_s = pt_df["static_baseline"].astype(float).values

    def _row(name, p):
        return {
            "method": name,
            "n":      int(len(p)),
            "mae":    round(float(mae(p, actual)), 1),
            "mape":   round(float(mape(p, actual)), 2),
            "w20r":   round(float(w20r(p, actual)), 2),
            "mean_actual":    int(round(float(np.mean(actual)))),
            "mean_pred":      int(round(float(np.mean(p)))),
            "median_actual":  int(round(float(np.median(actual)))),
            "median_pred":    int(round(float(np.median(p)))),
            "spearman":       round(float(_spearman(p, actual)), 3),
        }

    # Leave-one-out scale calibration: for each turn i, scale its
    # forecast by median(actual_{-i}) / median(pred_{-i}) computed
    # over all *other* turns. This estimates "what if we let the
    # deployer apply a single global scale factor learned from their
    # own historical traffic, with no per-turn leakage?".
    def _loo_calibrated(p):
        n = len(p)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            med_a = float(np.median(actual[mask]))
            med_p = float(np.median(p[mask])) or 1.0
            out[i] = p[i] * (med_a / med_p)
        return out

    pred_h_cal = _loo_calibrated(pred_h)
    pred_g_cal = _loo_calibrated(pred_g)
    pred_s_cal = _loo_calibrated(pred_s)

    summary = [
        _row("static_baseline",         pred_s),
        _row("static_baseline_cal",     pred_s_cal),
        _row("TABF_heuristic",          pred_h),
        _row("TABF_heuristic_cal",      pred_h_cal),
        _row("TABF_gbt",                pred_g),
        _row("TABF_gbt_cal",            pred_g_cal),
    ]

    # ---- Stage 2: TABF-driven optimizer simulation. ----
    # We treat each method's forecast as the proposed max_tokens cap
    # ContextOptimizer should target. We then count:
    #   over_budget_share: turns where the actual usage exceeded the
    #     forecast cap (would have been truncated or trigger overflow
    #     under a hard cap policy).
    #   slack_p50, slack_p95: 50th and 95th percentile of (forecast -
    #     actual) for turns within budget. Tighter is better.
    #   coverage_2x: share of turns where actual <= 2 * forecast (a
    #     loose "we did not blow up the budget by more than 2x" check
    #     that protects against hard outliers).
    pred_lookup = {
        "static_baseline":     pred_s,
        "static_baseline_cal": pred_s_cal,
        "TABF_heuristic":      pred_h,
        "TABF_heuristic_cal":  pred_h_cal,
        "TABF_gbt":            pred_g,
        "TABF_gbt_cal":        pred_g_cal,
    }
    for row in summary:
        m = row["method"]
        p = pred_lookup[m]
        over = actual > p
        slack = p - actual
        within_idx = np.where(actual <= p)[0]
        row.update({
            "over_budget_share": round(float(over.mean() * 100), 2),
            "slack_p50": int(round(float(np.percentile(slack[within_idx], 50)))) if len(within_idx) else None,
            "slack_p95": int(round(float(np.percentile(slack[within_idx], 95)))) if len(within_idx) else None,
            "coverage_2x": round(float((actual <= 2.0 * p).mean() * 100), 2),
        })

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(out_dir / "tabf_replay_summary.csv", index=False)

    print("\nStage 1+2 summary (forecast vs actual on real production turns):")
    print(sum_df.to_string(index=False))

    Path(out_dir / "tabf_replay_manifest.json").write_text(json.dumps({
        "n_turns": int(len(pt_df)),
        "static_baseline_value": int(round(median_total)),
        "notes": ("Stage 1 evaluates TABF as a pure forecaster against "
                  "actual production token usage. Stage 2 treats the "
                  "forecast as a max_tokens cap and reports over-budget "
                  "share + slack quantiles."),
    }, indent=2))


def _spearman(p, y):
    """Spearman rank correlation, no scipy dependency."""
    p_rank = pd.Series(p).rank().values
    y_rank = pd.Series(y).rank().values
    pm, ym = p_rank.mean(), y_rank.mean()
    num = ((p_rank - pm) * (y_rank - ym)).sum()
    den = np.sqrt(((p_rank - pm) ** 2).sum() * ((y_rank - ym) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--prod-csv", default="results/production_bq_turns.csv", type=Path)
    args = ap.parse_args()
    main(args.out, args.prod_csv)

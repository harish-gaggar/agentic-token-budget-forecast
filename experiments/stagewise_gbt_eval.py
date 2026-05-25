"""Compare total-token GBT + redistribution vs per-stage GBT regressors."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tabf.extractor import extract_signals
from tabf.classifier import classify, TASK_TYPES
from tabf.heuristic import HeuristicBudgetModel
from tabf.ml import GBTBudgetModel
from tabf.ml_stagewise import GBTBudgetModelStagewise, STAGES

from benchmark.corpus import build_corpus
from benchmark.simulator import simulate_dataset
from experiments.metrics import mae, mape, w20r

CORPUS_SEED, SIM_SEED, N_TASKS, TRAIN_FRAC = 1729, 20260514, 600, 0.7


def _split(rows, seed=42):
    rng = np.random.default_rng(seed)
    by_type: Dict[str, list] = {t: [] for t in TASK_TYPES}
    for r in rows:
        by_type[r["latent_type"]].append(r)
    train, test = [], []
    for lst in by_type.values():
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        sp = int(len(lst) * TRAIN_FRAC)
        train.extend(lst[i] for i in idx[:sp])
        test.extend(lst[i] for i in idx[sp:])
    return train, test


def _stage_w20r(model, test) -> Dict[str, float]:
    out = {}
    for s in STAGES:
        p, y = [], []
        for r in test:
            sig = extract_signals(r["text"])
            ktype, cx = classify(r["text"], sig)
            fc = model.forecast(ktype, cx, sig)
            p.append(getattr(fc, s))
            y.append(r["gt"][s])
        out[s] = w20r(np.asarray(p, float), np.asarray(y, float))
    return out


def _total_eval(model, test):
    preds, truths = [], []
    for r in test:
        sig = extract_signals(r["text"])
        ktype, cx = classify(r["text"], sig)
        preds.append(model.forecast(ktype, cx, sig).total_tokens)
        truths.append(r["gt"]["total"])
    p = np.asarray(preds, dtype=float)
    y = np.asarray(truths, dtype=float)
    return {"mae": mae(p, y), "mape": mape(p, y), "w20r": w20r(p, y)}


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = simulate_dataset(build_corpus(N_TASKS, seed=CORPUS_SEED), seed=SIM_SEED)
    train, test = _split(rows)

    def _traces(split):
        for r in split:
            sig = extract_signals(r["text"])
            ktype, cx = classify(r["text"], sig)
            yield sig, ktype, cx, r["gt"]

    heur = HeuristicBudgetModel()
    gbt_total = GBTBudgetModel(heuristic=heur).fit(
        [(s, k, c, g["input_tokens"], g["output"]) for s, k, c, g in _traces(train)]
    )
    gbt_stage = GBTBudgetModelStagewise(heuristic=heur).fit(list(_traces(train)))

    tot = _total_eval(gbt_total, test)
    stg = _total_eval(gbt_stage, test)
    sw_total = _stage_w20r(gbt_total, test)
    sw_stage = _stage_w20r(gbt_stage, test)

    rows_out = []
    for label, sw in [("TABF_gbt_redistribute", sw_total),
                      ("TABF_gbt_stagewise", sw_stage)]:
        for s in STAGES:
            rows_out.append({"model": label, "stage": s, "w20r": round(sw[s], 2)})
    pd.DataFrame(rows_out).to_csv(out_dir / "stagewise_gbt_comparison.csv", index=False)

    summary = pd.DataFrame([
        {"model": "TABF_gbt_redistribute", **{k: round(v, 2) for k, v in tot.items()}},
        {"model": "TABF_gbt_stagewise", **{k: round(v, 2) for k, v in stg.items()}},
        {"model": "TABF_gbt_redistribute", "metric": "mean_stage_w20r",
         "w20r": round(float(np.mean(list(sw_total.values()))), 2)},
        {"model": "TABF_gbt_stagewise", "metric": "mean_stage_w20r",
         "w20r": round(float(np.mean(list(sw_stage.values()))), 2)},
    ])
    summary.to_csv(out_dir / "stagewise_gbt_summary.csv", index=False)

    print("Total-token W20R:")
    print(f"  redistribute: {tot['w20r']:.2f}%")
    print(f"  stagewise:  {stg['w20r']:.2f}%")
    print("Mean per-stage W20R:")
    print(f"  redistribute: {np.mean(list(sw_total.values())):.2f}%")
    print(f"  stagewise:  {np.mean(list(sw_stage.values())):.2f}%")
    print("Per-stage W20R (stagewise model):")
    for s in STAGES:
        print(f"  {s:10s} {sw_stage[s]:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

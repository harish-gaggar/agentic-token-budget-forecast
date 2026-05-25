"""Oracle vs predicted labels on the synthetic test set.

Quantifies how much TABF gains from using the same rule chain that
constructed corpus latent labels (leakage concern).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from benchmark.corpus import build_corpus
from benchmark.simulator import simulate_dataset
from experiments.metrics import w20r
from tabf.classifier import classify
from tabf.extractor import extract_signals
from tabf.heuristic import HeuristicBudgetModel
from tabf.ml import GBTBudgetModel


CORPUS_SEED = 1729
SIM_SEED = 20260514
N_TASKS = 600
TRAIN_FRAC = 0.7


def _split(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(42)
    by_type: Dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["latent_type"], []).append(r)
    train, test = [], []
    for lst in by_type.values():
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        split = int(len(lst) * TRAIN_FRAC)
        train.extend(lst[i] for i in idx[:split])
        test.extend(lst[i] for i in idx[split:])
    return train, test


def _forecast_total(
    model,
    text: str,
    ktype: str,
    cx: str,
) -> int:
    sig = extract_signals(text)
    if isinstance(model, GBTBudgetModel):
        return model.forecast(ktype, cx, sig).total_tokens
    return model.forecast(ktype, cx, sig).total_tokens


def _eval_labels(model, rows: List[Dict], use_oracle: bool) -> float:
    preds, truth = [], []
    for r in rows:
        text = r["text"]
        if use_oracle:
            ktype = r["latent_type"]
            cx = r["latent_complexity_hint"]
        else:
            sig = extract_signals(text)
            ktype, cx = classify(text, sig)
        preds.append(_forecast_total(model, text, ktype, cx))
        truth.append(r["gt"]["total"])
    return w20r(np.array(preds), np.array(truth))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(N_TASKS, seed=CORPUS_SEED)
    rows = simulate_dataset(corpus, seed=SIM_SEED)
    train, test = _split(rows)

    heur = HeuristicBudgetModel()
    train_traces = []
    for r in train:
        sig = extract_signals(r["text"])
        ktype, cx = classify(r["text"], sig)
        train_traces.append((sig, ktype, cx, r["gt"]["input_tokens"], r["gt"]["output"]))
    gbt = GBTBudgetModel(heuristic=heur, random_state=0).fit(train_traces)

    rows_out = []
    for model_name, model in [("TABF heuristic", heur), ("TABF + GBT", gbt)]:
        pred_w = _eval_labels(model, test, use_oracle=False)
        oracle_w = _eval_labels(model, test, use_oracle=True)
        rows_out.append({
            "model": model_name,
            "w20r_predicted_labels": round(pred_w, 2),
            "w20r_oracle_labels": round(oracle_w, 2),
            "oracle_minus_predicted_pp": round(oracle_w - pred_w, 2),
        })

    df = pd.DataFrame(rows_out)
    path = out / "oracle_label_eval.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

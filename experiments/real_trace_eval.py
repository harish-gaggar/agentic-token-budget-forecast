"""Train and evaluate TABF on real production turns (83 with token metadata)."""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from tabf.extractor import extract_signals
from tabf.classifier import classify
from tabf.ml import _feature_vector
from experiments.metrics import mae, mape, w20r


def _task_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    for sep in ("\n---", "\n###", "\n```"):
        j = raw.find(sep)
        if j != -1:
            raw = raw[:j]
    return raw.strip()


def main(out_dir: Path, prod_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(prod_csv)
    df["task"] = df["input"].apply(_task_text)
    df = df[df["task"].str.len() > 0].copy()
    df["total_tokens"] = df["total_tokens"].astype(int)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_test = max(int(len(df) * 0.25), 15)
    test_idx = idx[:n_test]
    train = df.drop(test_idx)
    test = df.loc[test_idx]

    def _fit_predict(split_train, split_test, scale_from_train=True):
        X_tr = np.array([
            _feature_vector(extract_signals(t), *classify(t, extract_signals(t)))
            for t in split_train["task"]
        ], dtype=float)
        y_tr = split_train["total_tokens"].astype(float).values
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=6,
                                          learning_rate=0.06, random_state=0)
        m.fit(X_tr, y_tr)
        X_te = np.array([
            _feature_vector(extract_signals(t), *classify(t, extract_signals(t)))
            for t in split_test["task"]
        ], dtype=float)
        pred = np.maximum(m.predict(X_te), 1.0)
        actual = split_test["total_tokens"].astype(float).values
        if scale_from_train:
            scale = float(np.median(y_tr) / max(np.median(pred), 1.0))
            pred = pred * scale
        return pred, actual

    # Synthetic-pretrained is not available here; train on real train split only.
    pred, actual = _fit_predict(train, test, scale_from_train=True)
    row = {
        "split": "real_75_25",
        "n_train": len(train),
        "n_test": len(df),
        "mae": round(mae(pred, actual), 1),
        "mape": round(mape(pred, actual), 2),
        "w20r": round(w20r(pred, actual), 2),
        "calibration": "train_median_scale",
    }

    # Leave-one-out on full corpus (deployment calibration protocol)
    loo_p, actual_loo = [], []
    for i in range(len(df)):
        tr = df.drop(df.index[i])
        te = df.iloc[[i]]
        p, a = _fit_predict(tr, te, scale_from_train=True)
        loo_p.append(p[0])
        actual_loo.append(a[0])
    loo_p = np.asarray(loo_p)
    actual_te = np.asarray(actual_loo)
    row_loo = {
        "split": "real_loo_calibrated",
        "n_train": len(df) - 1,
        "n_test": len(df),
        "mae": round(mae(loo_p, actual_te), 1),
        "mape": round(mape(loo_p, actual_te), 2),
        "w20r": round(w20r(loo_p, actual_te), 2),
        "calibration": "loo_median_scale",
    }

    out = pd.DataFrame([row, row_loo])
    out.to_csv(out_dir / "real_trace_eval.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--prod-csv", default="results/production_bq_turns.csv", type=Path)
    main(ap.parse_args().out, ap.parse_args().prod_csv)

"""Calibrated confidence for TABF + GBT.

The reviewer pointed out:
  "Confidence estimation is under-specified; a single 'confidence in
   [0,1]' is mentioned, but calibration, coverage, and utility are
   not demonstrated."

We replace the constant 0.91 confidence currently emitted by GBT with
a data-driven interval-based confidence built from two extra quantile
regressors trained on the same input. For each task the quantile
models predict the 10th- and 90th-percentile total input tokens; the
half-width of that interval, normalised by the predicted mean,
becomes a per-prediction *interval-width score* in [0, 1], higher =
narrower = more confident.

We then evaluate:
  1. Coverage. How often does the actual ground-truth fall inside the
     predicted [q10, q90] interval? An honest predictor should cover
     close to 80% of the truth.
  2. Reliability. Bin predictions by interval-width score, plot
     within-20% rate inside each bin. A well-calibrated confidence
     should monotonically separate high-confidence (high W20R) from
     low-confidence (lower W20R) predictions.
  3. Expected calibration error (ECE) computed on the reliability
     bins above, as a single scalar summary.

Output:
  results/calibration_metrics.json
  results/fig_calibration_reliability.pdf
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from tabf.extractor import extract_signals
from tabf.classifier import classify, TASK_TYPES, COMPLEXITY_LEVELS
from tabf.heuristic import HeuristicBudgetModel
from tabf.ml import GBTBudgetModel, _feature_vector  # type: ignore

from benchmark.corpus import build_corpus
from benchmark.simulator import simulate_dataset

CORPUS_SEED = 1729
SIM_SEED    = 20260514
N_TASKS     = 600
TRAIN_FRAC  = 0.7

# Quantiles for the prediction interval.
LO_Q, HI_Q = 0.10, 0.90
NOMINAL_COVERAGE = HI_Q - LO_Q  # 0.80


def _feat(text: str):
    sig = extract_signals(text)
    ktype, cx = classify(text, sig)
    return _feature_vector(sig, ktype, cx)


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

    # Fit the canonical GBT for point predictions (same as run_all).
    heur = HeuristicBudgetModel()
    train_traces = []
    for r in train:
        sig = extract_signals(r["text"])
        ktype, cx = classify(r["text"], sig)
        train_traces.append((sig, ktype, cx, r["gt"]["input_tokens"], r["gt"]["output"]))
    gbt = GBTBudgetModel(heuristic=heur, random_state=0).fit(train_traces)

    # Fit two extra quantile GBTs on (features -> total tokens).
    X_train = np.asarray([_feat(r["text"]) for r in train], dtype=float)
    y_train = np.asarray([r["gt"]["total"] for r in train], dtype=float)

    # sklearn's GradientBoostingRegressor with loss='quantile' learns
    # asymmetric pinball loss to estimate a chosen quantile.
    q_lo = GradientBoostingRegressor(
        loss="quantile", alpha=LO_Q, n_estimators=200, max_depth=4,
        learning_rate=0.06, random_state=0,
    ).fit(X_train, y_train)
    q_hi = GradientBoostingRegressor(
        loss="quantile", alpha=HI_Q, n_estimators=200, max_depth=4,
        learning_rate=0.06, random_state=0,
    ).fit(X_train, y_train)

    # Test-set predictions.
    X_test = np.asarray([_feat(r["text"]) for r in test], dtype=float)
    y_test = np.asarray([r["gt"]["total"] for r in test], dtype=float)
    p_mean = np.array([gbt.forecast(*_classify_for(r["text"])).total_tokens
                       for r in test], dtype=float)
    lo = q_lo.predict(X_test)
    hi = q_hi.predict(X_test)
    # Sanity: enforce lo <= hi (quantile models can occasionally cross).
    lo = np.minimum(lo, hi)
    hi = np.maximum(lo, hi)

    # 1. Coverage of the nominal 80% interval.
    inside = (y_test >= lo) & (y_test <= hi)
    coverage = float(inside.mean())

    # 2. Per-prediction interval-width score in [0, 1].
    #    width / mean   -> smaller is tighter.
    #    confidence = exp(-(width / mean))   -> bounded in (0, 1].
    width = hi - lo
    width_over_mean = width / np.clip(p_mean, 1.0, None)
    conf = np.exp(-width_over_mean)

    # 3. Reliability diagram: bin by confidence, plot empirical W20R.
    n_bins = 10
    bins = np.linspace(conf.min(), conf.max(), n_bins + 1)
    err = np.abs(p_mean - y_test) / np.clip(y_test, 1.0, None)
    hit_20 = (err <= 0.20).astype(float)

    rel_rows = []
    for i in range(n_bins):
        lo_b, hi_b = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo_b) & (conf <= hi_b)
        else:
            mask = (conf >= lo_b) & (conf < hi_b)
        if mask.sum() == 0:
            continue
        rel_rows.append({
            "bin": i,
            "conf_lo": float(lo_b),
            "conf_hi": float(hi_b),
            "n": int(mask.sum()),
            "mean_conf": float(conf[mask].mean()),
            "w20r": float(hit_20[mask].mean() * 100),
        })

    # Expected calibration error vs the ideal "confidence == W20R".
    ece = 0.0
    n_total = len(conf)
    for row in rel_rows:
        ece += (row["n"] / n_total) * abs(row["mean_conf"] - row["w20r"] / 100.0)

    # 4. Coverage at multiple nominal levels (re-fit additional quantile
    #    pairs for 50% and 90% intervals so we can report a table).
    coverage_table = [{
        "nominal_level": int(round(NOMINAL_COVERAGE * 100)),
        "empirical_coverage_pct": round(coverage * 100, 2),
    }]
    for alpha_lo, alpha_hi in [(0.25, 0.75), (0.05, 0.95)]:
        m_lo = GradientBoostingRegressor(
            loss="quantile", alpha=alpha_lo, n_estimators=200,
            max_depth=4, learning_rate=0.06, random_state=0,
        ).fit(X_train, y_train)
        m_hi = GradientBoostingRegressor(
            loss="quantile", alpha=alpha_hi, n_estimators=200,
            max_depth=4, learning_rate=0.06, random_state=0,
        ).fit(X_train, y_train)
        plo = m_lo.predict(X_test)
        phi = m_hi.predict(X_test)
        plo, phi = np.minimum(plo, phi), np.maximum(plo, phi)
        cov = float(((y_test >= plo) & (y_test <= phi)).mean())
        coverage_table.append({
            "nominal_level": int(round((alpha_hi - alpha_lo) * 100)),
            "empirical_coverage_pct": round(cov * 100, 2),
        })

    # Sort coverage_table by nominal level ascending for nicer reading.
    coverage_table.sort(key=lambda r: r["nominal_level"])

    metrics = {
        "n_train":           int(len(train)),
        "n_test":            int(len(test)),
        "interval":          {"lo_q": LO_Q, "hi_q": HI_Q,
                              "nominal_coverage_pct": int(round(NOMINAL_COVERAGE * 100)),
                              "empirical_coverage_pct": round(coverage * 100, 2)},
        "expected_calibration_error": round(float(ece), 4),
        "reliability_bins":  rel_rows,
        "coverage_table":    coverage_table,
        "method": "quantile_gbt",
        "notes": ("Confidence = exp(-(interval_width / point_pred)). "
                  "10 equal-width bins on [min(conf), max(conf)]. ECE "
                  "is mean |mean_conf - empirical_W20R| weighted by bin "
                  "occupancy."),
    }
    Path(out_dir / "calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    # Console summary.
    print(f"Coverage of nominal {int(NOMINAL_COVERAGE*100)}% interval: "
          f"{coverage*100:.2f}%   (target 80.0%)")
    print(f"Expected calibration error: {ece:.4f}")
    print("Coverage at multiple nominal levels:")
    for row in coverage_table:
        print(f"  nominal {row['nominal_level']:>3d}%   empirical "
              f"{row['empirical_coverage_pct']:>5.2f}%")
    print("Reliability bins (n, mean_conf, empirical W20R%):")
    for row in rel_rows:
        print(f"  bin {row['bin']:>2d}  n={row['n']:>3d}  "
              f"conf={row['mean_conf']:.3f}  W20R={row['w20r']:.2f}%")

    # Plot reliability diagram.
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    mean_confs = [r["mean_conf"] for r in rel_rows]
    w20rs      = [r["w20r"] / 100.0 for r in rel_rows]
    sizes      = [40 + 6 * r["n"] for r in rel_rows]
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=0.8,
            label="Perfect calibration")
    ax.scatter(mean_confs, w20rs, s=sizes, color="#1a365d",
               edgecolor="black", linewidth=0.5, zorder=3,
               label="TABF + GBT (bin size $\\propto$ marker)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted confidence (interval-width score)")
    ax.set_ylabel("Empirical within-20% rate")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_calibration_reliability.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out_dir/'calibration_metrics.json'} and "
          f"{out_dir/'fig_calibration_reliability.pdf'}")


def _classify_for(text: str):
    sig = extract_signals(text)
    ktype, cx = classify(text, sig)
    return (ktype, cx, sig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

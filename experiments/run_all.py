"""Run the full TABF evaluation: build the corpus, simulate ground-truth
traces, evaluate baselines + heuristic + GBT, run ablations, and write
CSV results plus PDF figures.

Usage:
    python -m experiments.run_all --out results/
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabf.extractor import extract_signals, Signals
from tabf.classifier import classify, TASK_TYPES, COMPLEXITY_LEVELS
from tabf.heuristic import HeuristicBudgetModel
from tabf.ml import GBTBudgetModel

from benchmark.corpus import build_corpus, save_corpus
from benchmark.simulator import simulate_dataset

from experiments.baselines import (
    UniformBaseline, TypeOnlyHeuristic, LengthProportional,
    RidgeOnFeatures, RandomForestOnFeatures, HistGBOnFeatures,
    RidgeOnTfidf, HistGBOnTfidf,
)
from experiments.metrics import mae, mape, w20r, mcnemar_p


CORPUS_SEED = 1729
SIM_SEED = 20260514     # matches today's date for reproducibility
N_TASKS = 600
TRAIN_FRAC = 0.7


# ---------- core eval helpers ------------------------------------------------

def _predict_one(model, task: str) -> int:
    if hasattr(model, "forecast"):  # TABF heuristic or ml-backed
        sig = extract_signals(task)
        ktype, cx = classify(task, sig)
        if model.__class__.__name__ == "GBTBudgetModel":
            fc = model.forecast(ktype, cx, sig)
        else:
            fc = model.forecast(ktype, cx, sig)
        return fc.total_tokens
    return model.predict(task).total


def _evaluate(model, rows: List[Dict]) -> Dict:
    preds = np.array([_predict_one(model, r["text"]) for r in rows], dtype=float)
    truth = np.array([r["gt"]["total"] for r in rows], dtype=float)
    hits = (np.abs(preds - truth) / np.clip(truth, 1.0, None)) <= 0.20
    return {
        "preds": preds, "truth": truth, "hits": hits,
        "mae": mae(preds, truth),
        "mape": mape(preds, truth),
        "w20r": w20r(preds, truth),
    }


# ---------- main pipeline ----------------------------------------------------

def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build the corpus + ground-truth traces.
    corpus = build_corpus(N_TASKS, seed=CORPUS_SEED)
    save_corpus(corpus, out_dir / "corpus.json")
    rows = simulate_dataset(corpus, seed=SIM_SEED)
    Path(out_dir / "traces.json").write_text(json.dumps(rows, indent=2))

    # Train / test split (stratified by latent task type).
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

    n_train, n_test = len(train), len(test)
    print(f"Corpus n={len(rows)}  train={n_train}  test={n_test}")

    # 2) Fit baselines that need training. Naive baselines first,
    #    then the learned ones requested by the reviewer (Ridge, RF,
    #    HistGB on hand features; Ridge + HistGB on TF-IDF bag of words).
    train_texts = [r["text"] for r in train]
    train_totals = [r["gt"]["total"] for r in train]

    uniform = UniformBaseline().fit(train_totals)
    length = LengthProportional().fit(
        [len(r["text"].split()) for r in train], train_totals,
    )
    type_only = TypeOnlyHeuristic()
    heur = HeuristicBudgetModel()

    ridge_hand   = RidgeOnFeatures().fit(train_texts, train_totals)
    rf_hand      = RandomForestOnFeatures().fit(train_texts, train_totals)
    histgb_hand  = HistGBOnFeatures().fit(train_texts, train_totals)
    ridge_tfidf  = RidgeOnTfidf().fit(train_texts, train_totals)
    histgb_tfidf = HistGBOnTfidf().fit(train_texts, train_totals)

    # 3) Fit the GBT model on train traces.
    train_traces = []
    for r in train:
        sig = extract_signals(r["text"])
        ktype, cx = classify(r["text"], sig)
        train_traces.append((sig, ktype, cx, r["gt"]["input_tokens"], r["gt"]["output"]))
    gbt = GBTBudgetModel(heuristic=heur, random_state=0).fit(train_traces)

    # 4) Main results on the held-out test set.
    results = {}
    for name, model in [
        ("B1_uniform",       uniform),
        ("B2_type_only",     type_only),
        ("B3_length",        length),
        ("B4_ridge_hand",    ridge_hand),
        ("B5_rf_hand",       rf_hand),
        ("B6_histgb_hand",   histgb_hand),
        ("B7_ridge_tfidf",   ridge_tfidf),
        ("B8_histgb_tfidf",  histgb_tfidf),
        ("TABF_heuristic",   heur),
        ("TABF_gbt",         gbt),
    ]:
        results[name] = _evaluate(model, test)
        print(f"{name:18s}  MAE={results[name]['mae']:7.1f}  "
              f"MAPE={results[name]['mape']:5.2f}%  "
              f"W20R={results[name]['w20r']:5.2f}%")

    # Save main results CSV.
    main_df = pd.DataFrame([
        {"method": k,
         "mae": round(v["mae"], 1),
         "mape": round(v["mape"], 2),
         "w20r": round(v["w20r"], 2)}
        for k, v in results.items()
    ])
    main_df.to_csv(out_dir / "main_results.csv", index=False)

    # 5) McNemar significance: TABF+GBT vs each baseline on W20R hits.
    sig_rows = []
    a = results["TABF_gbt"]["hits"]
    for name in [
        "B1_uniform", "B2_type_only", "B3_length",
        "B4_ridge_hand", "B5_rf_hand", "B6_histgb_hand",
        "B7_ridge_tfidf", "B8_histgb_tfidf",
        "TABF_heuristic",
    ]:
        sig_rows.append({"vs": name, "p_value": mcnemar_p(a, results[name]["hits"])})
    pd.DataFrame(sig_rows).to_csv(out_dir / "significance.csv", index=False)
    for row in sig_rows:
        print(f"McNemar TABF_gbt vs {row['vs']:18s}  p={row['p_value']:.5f}")

    # 6) Stratified results for TABF_gbt by latent task type.
    strat_rows = []
    for t in TASK_TYPES:
        mask = np.array([r["latent_type"] == t for r in test])
        if mask.sum() == 0:
            continue
        p = results["TABF_gbt"]["preds"][mask]
        y = results["TABF_gbt"]["truth"][mask]
        strat_rows.append({
            "task_type": t,
            "n": int(mask.sum()),
            "mae": round(mae(p, y), 1),
            "mape": round(mape(p, y), 2),
            "w20r": round(w20r(p, y), 2),
            "avg_gt": int(round(np.mean(y))),
            "avg_pred": int(round(np.mean(p))),
        })
    pd.DataFrame(strat_rows).to_csv(out_dir / "stratified_results.csv", index=False)

    # 7) Ablation. Each row drops one feature *end-to-end* so the
    # downstream classifier no longer sees it.
    def _ablate_sig(sig: Signals, drop: str) -> Signals:
        return Signals(
            word_count=sig.word_count,
            has_multi_step=False if drop == "multi_step" else sig.has_multi_step,
            has_tool_keywords=False if drop == "tool_keywords" else sig.has_tool_keywords,
            has_code_keywords=False if drop == "code_keywords" else sig.has_code_keywords,
            has_file_refs=False if drop == "file_refs" else sig.has_file_refs,
            has_comparison=False if drop == "comparison" else sig.has_comparison,
            question_count=sig.question_count,
            entity_count=0 if drop == "entity_count" else sig.entity_count,
        )

    def _ablate_predict(text: str, drop: str, *,
                        flat_complexity: bool, no_word_adj: bool) -> int:
        sig = extract_signals(text)
        sig = _ablate_sig(sig, drop) if drop else sig
        ktype, cx = classify(text, sig)
        if flat_complexity:
            cx = "medium"
        h = HeuristicBudgetModel(word_adj_per_word=0.0 if no_word_adj else 0.008)
        return h.forecast(ktype, cx, sig).total_tokens

    def _ablate_eval(name: str, drop: str, *,
                     flat_complexity=False, no_word_adj=False):
        preds = np.array([_ablate_predict(r["text"], drop,
                                          flat_complexity=flat_complexity,
                                          no_word_adj=no_word_adj)
                          for r in test], dtype=float)
        truth = np.array([r["gt"]["total"] for r in test], dtype=float)
        return {
            "config": name,
            "mae": round(mae(preds, truth), 1),
            "mape": round(mape(preds, truth), 2),
            "w20r": round(w20r(preds, truth), 2),
        }

    abl_rows = [{
        "config": "full_heuristic",
        "mae": round(results["TABF_heuristic"]["mae"], 1),
        "mape": round(results["TABF_heuristic"]["mape"], 2),
        "w20r": round(results["TABF_heuristic"]["w20r"], 2),
    }]
    abl_rows.append(_ablate_eval("no_complexity_scaling", drop=None,
                                 flat_complexity=True))
    for f in ["multi_step", "tool_keywords", "code_keywords",
              "file_refs", "comparison", "entity_count"]:
        abl_rows.append(_ablate_eval(f"no_{f}", drop=f))
    abl_rows.append(_ablate_eval("no_word_adjustment", drop=None, no_word_adj=True))
    for row in abl_rows:
        print(f"ablation {row['config']:24s}  MAE={row['mae']:.1f}  "
              f"MAPE={row['mape']:.2f}%  W20R={row['w20r']:.2f}%")
    pd.DataFrame(abl_rows).to_csv(out_dir / "ablation_results.csv", index=False)

    # 7b) Classifier agreement with latent labels.
    type_hits, cx_hits, both_hits = 0, 0, 0
    for r in test:
        sig = extract_signals(r["text"])
        kt, cx = classify(r["text"], sig)
        if kt == r["latent_type"]:
            type_hits += 1
        if cx == r["latent_complexity_hint"]:
            cx_hits += 1
        if kt == r["latent_type"] and cx == r["latent_complexity_hint"]:
            both_hits += 1
    classifier_acc = {
        "task_type_acc": round(100.0 * type_hits / len(test), 2),
        "complexity_acc": round(100.0 * cx_hits / len(test), 2),
        "both_acc": round(100.0 * both_hits / len(test), 2),
    }
    Path(out_dir / "classifier_accuracy.json").write_text(
        json.dumps(classifier_acc, indent=2)
    )
    print(f"classifier task_type acc = {classifier_acc['task_type_acc']:.1f}%  "
          f"complexity acc = {classifier_acc['complexity_acc']:.1f}%  "
          f"both acc = {classifier_acc['both_acc']:.1f}%")

    # 8) Figures.
    _plot_w20r_bars(results, out_dir / "fig_w20r.pdf")
    _plot_error_distribution(results, out_dir / "fig_error_dist.pdf")
    _plot_calibration(results["TABF_gbt"], out_dir / "fig_calibration.pdf")
    _plot_per_type_w20r(strat_rows, out_dir / "fig_per_type_w20r.pdf")

    # 9) Manifest with all numbers for direct embedding in the paper.
    manifest = {
        "n_corpus": len(rows),
        "n_train": n_train,
        "n_test": n_test,
        "corpus_seed": CORPUS_SEED,
        "sim_seed": SIM_SEED,
        "main": {k: {kk: float(v[kk]) for kk in ("mae", "mape", "w20r")}
                 for k, v in results.items()},
        "significance": {row["vs"]: row["p_value"] for row in sig_rows},
        "stratified": strat_rows,
        "ablations": abl_rows,
        "classifier": classifier_acc,
    }
    Path(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nWrote results to", out_dir)


# ---------- figures ----------------------------------------------------------

_NAMES = {
    "B1_uniform":      "B1: Uniform",
    "B2_type_only":    "B2: Type-only",
    "B3_length":       "B3: Length-prop.",
    "B4_ridge_hand":   "B4: Ridge (hand)",
    "B5_rf_hand":      "B5: RF (hand)",
    "B6_histgb_hand":  "B6: HistGB (hand)",
    "B7_ridge_tfidf":  "B7: Ridge (TF-IDF)",
    "B8_histgb_tfidf": "B8: HistGB (TF-IDF)",
    "TABF_heuristic":  "TABF Heuristic",
    "TABF_gbt":        "TABF + GBT",
}

# Short x-axis labels for the W20R bar chart (full names stay in the caption/table).
_W20R_XLABELS = {
    "B1_uniform":      "B1\nUniform",
    "B2_type_only":    "B2\nType",
    "B3_length":       "B3\nLength",
    "B4_ridge_hand":   "B4\nRidge",
    "B5_rf_hand":      "B5\nRF",
    "B6_histgb_hand":  "B6\nHistGB",
    "B7_ridge_tfidf":  "B7\nRidge-T",
    "B8_histgb_tfidf": "B8\nHistGB-T",
    "TABF_heuristic":  "TABF\nheur.",
    "TABF_gbt":        "TABF\n+GBT",
}

_W20R_COLOURS = {
    "B1_uniform": "#9aa0a6",
    "B2_type_only": "#9aa0a6",
    "B3_length": "#9aa0a6",
    "B4_ridge_hand": "#718096",
    "B5_rf_hand": "#1a365d",
    "B6_histgb_hand": "#718096",
    "B7_ridge_tfidf": "#718096",
    "B8_histgb_tfidf": "#718096",
    "TABF_heuristic": "#2b6cb0",
    "TABF_gbt": "#1a365d",
}


def _plot_w20r_bars(results, path):
    keys = list(results.keys())
    labels = [_W20R_XLABELS[k] for k in keys]
    vals = [results[k]["w20r"] for k in keys]
    colours = [_W20R_COLOURS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(keys))
    bars = ax.bar(
        x, vals, color=colours, edgecolor="black", linewidth=0.4, width=0.72,
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.2,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, linespacing=0.95)
    ax.set_ylabel("Within-20% rate (%)")
    ax.set_ylim(0, max(vals) + 14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.22, left=0.10, right=0.98, top=0.92)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_error_distribution(results, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for k, c in [("B1_uniform", "#bbb"), ("B2_type_only", "#888"),
                 ("TABF_heuristic", "#2b6cb0"), ("TABF_gbt", "#1a365d")]:
        err = (results[k]["preds"] - results[k]["truth"]) / results[k]["truth"]
        ax.hist(err * 100, bins=40, alpha=0.55, label=_NAMES[k],
                color=c, edgecolor="black", linewidth=0.3)
    ax.axvspan(-20, 20, color="#fef3c7", alpha=0.5, zorder=0,
               label=u"\u00b120% band")
    ax.set_xlabel("Signed percentage error (%)")
    ax.set_ylabel("Tasks")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.set_xlim(-120, 120)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(gbt_res, path):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    p, y = gbt_res["preds"], gbt_res["truth"]
    ax.scatter(y, p, s=10, alpha=0.5, color="#1a365d", edgecolor="none")
    lim = float(max(p.max(), y.max())) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#888", linewidth=0.8)
    ax.fill_between([0, lim], [0, 0.8 * lim], [0, 1.2 * lim],
                    color="#fef3c7", alpha=0.6, zorder=0)
    ax.set_xlabel("Ground truth tokens")
    ax.set_ylabel("Predicted tokens (TABF + GBT)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_per_type_w20r(strat_rows, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    rows = sorted(strat_rows, key=lambda r: r["w20r"], reverse=True)
    labels = [r["task_type"].replace("_", " ") for r in rows]
    vals = [r["w20r"] for r in rows]
    ax.barh(labels, vals, color="#2b6cb0", edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Within-20% rate (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

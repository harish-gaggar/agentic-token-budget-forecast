"""Production quality outcomes joined to turn-level telemetry."""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(out_dir: Path, turns_csv: Path, quality_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    turns = pd.read_csv(turns_csv)
    qual = pd.read_csv(quality_csv)

    # Wide quality table per request_id
    wide = qual.pivot_table(
        index="request_id", columns="metric_name", values="result", aggfunc="first"
    ).reset_index()

    merged = turns.merge(wide, on="request_id", how="left")
    merged.to_csv(out_dir / "production_turns_with_quality.csv", index=False)

    def _rate(col):
        if col not in merged.columns:
            return None
        s = pd.to_numeric(merged[col], errors="coerce")
        return float(s.mean()) if s.notna().any() else None

    status_ok = merged["status"].eq("complete").mean() * 100
    tsr = _rate("task_success_rate")

    rows = [
        {"metric": "turns_n", "value": len(merged)},
        {"metric": "status_success_pct", "value": round(status_ok, 2)},
        {"metric": "eval_task_success_rate_mean", "value": round(tsr * 100, 2) if tsr else None},
        {"metric": "median_latency_ms_complete",
         "value": float(merged.loc[merged["status"] == "complete", "duration_ms"].median())
         if "duration_ms" in merged and (merged["status"] == "complete").any() else None},
        {"metric": "median_latency_ms_error",
         "value": float(merged.loc[merged["status"] == "error", "duration_ms"].median())
         if "duration_ms" in merged and (merged["status"] == "error").any() else None},
        {"metric": "median_input_tokens_complete",
         "value": float(merged.loc[merged["status"] == "complete", "input_tokens"].median())
         if (merged["status"] == "complete").any() else None},
        {"metric": "median_input_tokens_error",
         "value": float(merged.loc[merged["status"] == "error", "input_tokens"].median())
         if (merged["status"] == "error").any() else None},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "production_quality_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--turns", default="results/production_bq_turns.csv", type=Path)
    ap.add_argument("--quality", default="results/production_quality_join.csv", type=Path)
    main(ap.parse_args().out, ap.parse_args().turns, ap.parse_args().quality)

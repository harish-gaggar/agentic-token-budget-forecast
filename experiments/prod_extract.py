"""Pull the latest production tracking + quality data from BigQuery.

The reviewer flagged that ContextOptimizer's production evidence is
narrow (N=14 directly observed LLM calls + 75 turn-level rows). Since
that prior pull, the tracking dataset has grown:

  agent_job_tracking            75 -> 139 turns      (+85%)
  agent_quality_eval            12 -> 1102 rows
  agent_eval_scores             12 -> 137
  agent_quality_metrics_detail  12 -> 134

This script re-extracts the latest population so the paper can quote
the larger sample. We also join quality scores onto turn rows so we
can show that high-savings turns are not over-represented among
quality regressions.

Outputs (overwritten):
  results/production_bq_turns.csv         one row per turn (139 rows)
  results/production_bq_population_stats.csv   rolled-up population stats
  results/production_quality_join.csv     turn x quality dimension long
  results/production_extract_manifest.json  what we pulled, when, etc.
"""

from __future__ import annotations
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT = "prod-ck-analytics-data-99"
DATASET = "genai_agent_eval_tracking"
AGENT   = "bigquery-langgraph-agent"


def _run_bq(sql: str) -> pd.DataFrame:
    """Run a bq query and return a DataFrame.

    bq's --format=csv preserves quoting properly for free-text columns,
    which we need because `input` contains multi-line SQL.
    """
    out = subprocess.run(
        ["bq", "query",
         f"--project_id={PROJECT}",
         "--use_legacy_sql=false",
         "--format=csv",
         "--max_rows=10000",
         sql],
        capture_output=True, text=True, check=True,
    )
    # bq prepends a "Waiting on bqjob..." line; drop anything before
    # the header.
    lines = out.stdout.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if "," in ln and not ln.startswith("Waiting")),
                      0)
    import io
    csv_text = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(csv_text))


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Turn-level token data ----
    turns_sql = f"""
SELECT
  request_id,
  ts,
  input,
  status,
  CAST(NULLIF((SELECT value FROM UNNEST(metadata) WHERE key='input_tokens'), '') AS INT64)  AS input_tokens,
  CAST(NULLIF((SELECT value FROM UNNEST(metadata) WHERE key='output_tokens'), '') AS INT64) AS output_tokens,
  CAST(NULLIF((SELECT value FROM UNNEST(metadata) WHERE key='total_tokens'), '') AS INT64)  AS total_tokens,
  CAST(NULLIF((SELECT value FROM UNNEST(metadata) WHERE key='llm_call_count'), '') AS INT64)  AS llm_calls,
  CAST(NULLIF((SELECT value FROM UNNEST(metadata) WHERE key='tool_call_count'), '') AS INT64) AS tool_calls,
  (SELECT value FROM UNNEST(metadata) WHERE key='llm_model') AS model_id_meta,
  duration_ms
FROM `{PROJECT}.{DATASET}.agent_job_tracking`
WHERE agent_name = '{AGENT}'
"""
    turns = _run_bq(turns_sql)

    # Keep only turns that actually have token data (a few may be NULL
    # for failed runs).
    turns = turns[turns["input_tokens"].notna()].copy()
    turns["input_tokens"]  = turns["input_tokens"].astype("Int64")
    turns["output_tokens"] = turns["output_tokens"].astype("Int64")
    turns["total_tokens"]  = turns["total_tokens"].astype("Int64")
    turns["llm_calls"]     = turns["llm_calls"].astype("Int64")
    turns["tool_calls"]    = turns["tool_calls"].astype("Int64")

    turns.to_csv(out_dir / "production_bq_turns.csv", index=False)
    print(f"production_bq_turns.csv  n={len(turns)}")

    # ---- 2. Rolled-up population stats ----
    pop = {
        "extracted_at_utc":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turns_with_token_metadata": int(len(turns)),
        "total_input_tokens":        int(turns["input_tokens"].sum()),
        "total_output_tokens":       int(turns["output_tokens"].sum()),
        "total_tokens":              int(turns["total_tokens"].sum()),
        "total_llm_calls":           int(turns["llm_calls"].fillna(0).sum()),
        "total_tool_calls":          int(turns["tool_calls"].fillna(0).sum()),
        "total_duration_seconds":    round(float(turns["duration_ms"].fillna(0).sum()) / 1000.0, 1),
        "llm_calls_per_turn_mean":   round(float(turns["llm_calls"].fillna(0).mean()), 2),
        "llm_calls_per_turn_median": float(turns["llm_calls"].fillna(0).median()),
        "llm_calls_per_turn_p95":    float(turns["llm_calls"].fillna(0).quantile(0.95)),
        "llm_calls_per_turn_max":    int(turns["llm_calls"].fillna(0).max()),
        "input_tokens_per_turn_mean":   round(float(turns["input_tokens"].mean()), 1),
        "input_tokens_per_turn_median": float(turns["input_tokens"].median()),
        "input_tokens_per_turn_p95":    float(turns["input_tokens"].quantile(0.95)),
        "input_tokens_per_turn_max":    int(turns["input_tokens"].max()),
        "tool_calls_per_turn_mean":   round(float(turns["tool_calls"].fillna(0).mean()), 2),
        "tool_calls_per_turn_median": float(turns["tool_calls"].fillna(0).median()),
        "tool_calls_per_turn_p95":    float(turns["tool_calls"].fillna(0).quantile(0.95)),
        "tool_calls_per_turn_max":    int(turns["tool_calls"].fillna(0).max()),
    }

    pop_df = pd.DataFrame(list(pop.items()), columns=["metric", "value"])
    pop_df.to_csv(out_dir / "production_bq_population_stats.csv", index=False)
    print("Rolled-up population stats:")
    for k, v in pop.items():
        print(f"  {k:32s} {v}")

    # ---- 3. Quality eval (long form: one row per turn x metric) ----
    qual_sql = f"""
SELECT
  request_id,
  metric_name,
  result,
  ts
FROM `{PROJECT}.{DATASET}.agent_quality_eval`
WHERE agent_name = '{AGENT}'
"""
    qual = _run_bq(qual_sql)
    qual.to_csv(out_dir / "production_quality_join.csv", index=False)
    print(f"production_quality_join.csv  n={len(qual)}")

    # ---- 4. Manifest ----
    Path(out_dir / "production_extract_manifest.json").write_text(json.dumps({
        "extracted_at_utc": pop["extracted_at_utc"],
        "project":          PROJECT,
        "dataset":          DATASET,
        "agent_name":       AGENT,
        "turns_n":          int(len(turns)),
        "quality_n":        int(len(qual)),
        "notes": ("Pulled by experiments/prod_extract.py. Replaces the "
                  "older snapshot (75 turns, 12 quality rows) with the "
                  "latest population."),
    }, indent=2))
    print(f"\nWrote {out_dir/'production_bq_turns.csv'}, "
          f"{out_dir/'production_bq_population_stats.csv'}, "
          f"{out_dir/'production_quality_join.csv'}, "
          f"and manifest.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", type=Path)
    main(ap.parse_args().out)

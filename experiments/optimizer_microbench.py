"""Microbenchmark for the ContextOptimizer pipeline (Sections 7-8 of paper.tex).

Runs the agent.context_optimizer pipeline against synthetic LangChain
message lists whose shape brackets the production input distribution,
and writes two CSVs:

* per-call wall-clock overhead (mean, median, p95, max)
* per-layer token attribution (which layers save tokens)
* compression ratio across scenarios of varying tool-result size and
  history depth

Inputs are synthetic but the timings and per-layer counts are real
measurements of the running pipeline. The script makes no LLM call and
needs no network access; all token counts come from tiktoken, the same
counter the production code uses.

Usage (run inside a container that has the optimizer + langchain_core
installed)::

    docker cp experiments/optimizer_microbench.py \
        de_genai-agent-template-genai-agent-template-1:/tmp/
    docker exec de_genai-agent-template-genai-agent-template-1 \
        python /tmp/optimizer_microbench.py --out /tmp/microbench.csv \
                                            --layers-out /tmp/microbench_layers.csv
    docker cp de_genai-agent-template-genai-agent-template-1:/tmp/microbench.csv \
        results/optimizer_microbench.csv
    docker cp de_genai-agent-template-genai-agent-template-1:/tmp/microbench_layers.csv \
        results/optimizer_microbench_layers.csv

The two CSVs feed the overhead and per-layer tables in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---- Fail fast if the optimizer or LangChain are not present -----------------
from agent.context_optimizer import presets
from agent.context_optimizer.adapters import langchain_adapter as lc
from agent.context_optimizer.core.tokens import count_items_tokens
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# ----- Fixtures
# Static text snippets used to build synthetic conversations. None of
# this is real user data; the snippets are deterministic fixtures that
# match the rough shape of what the production BigQuery agent sees so
# that the optimizer's behaviour stays measurable. The bulk of the
# token weight lives inside ToolMessage payloads, which is what the
# production traces also show.

SYSTEM_PROMPT = (
    "You are a careful data assistant. Use the available tools to answer the "
    "user's question, prefer SQL aggregation over fetching raw rows, and "
    "always cite the table name. Tables: revenue_by_vertical, signups, "
    "monthly_revenue, churn_events. Respond with at most two short paragraphs "
    "and a single fenced code block when relevant."
)

USER_TURNS = [
    "Show revenue by vertical for the last 30 days, ordered descending.",
    "What was the weekly trend of signups over the past 8 weeks?",
    "Compare churn this quarter vs the previous quarter, by segment.",
    "Top 10 customers by lifetime value, with their segment.",
    "Daily revenue for the last 14 days, with a 7-day rolling average.",
]


def _synth_query_result(rows: int, cols: int = 8, *, seed: int) -> str:
    """Build a fake BigQuery JSON result of approximately the requested row count.

    Token weight scales roughly linearly with ``rows`` because each row carries
    the same JSON schema overhead.
    """
    rng = random.Random(seed)
    keys = [f"col_{i}" for i in range(cols)]
    out = []
    for r in range(rows):
        out.append({k: f"value_{r}_{k}_{rng.randint(0, 1_000_000)}" for k in keys})
    # Wrap exactly the way the production tool wrapper does, so token
    # counts match what production sees.
    return json.dumps({"rows": out, "schema": keys, "row_count": rows}, indent=2)


def _scenario(
    *, tool_calls: int, rows_per_call: int, seed: int
) -> List[Any]:
    """Build one LangChain message list of the shape the production agent sees.

    The shape is fixed: system prompt, user query, ``tool_calls`` rounds
    of ``AIMessage(tool_calls=...) -> ToolMessage``, then a final
    ``HumanMessage`` standing in for the user's follow-up. The tool
    result payload size is controlled by ``rows_per_call``.
    """
    rng = random.Random(seed)
    msgs: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_TURNS[rng.randrange(len(USER_TURNS))]),
    ]
    for i in range(tool_calls):
        tc_id = f"call_{seed}_{i}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "id": tc_id,
                    "name": "query_warehouse",
                    "args": {"sql": f"SELECT * FROM table_{i} LIMIT {rows_per_call}"},
                }],
            )
        )
        msgs.append(
            ToolMessage(
                content=_synth_query_result(rows_per_call, seed=seed * 1000 + i),
                tool_call_id=tc_id,
                name="query_warehouse",
            )
        )
    # Final follow-up turn (the optimizer always sees one new HumanMessage at
    # the end, pinned by the adapter).
    msgs.append(HumanMessage(content="Now summarise the findings briefly."))
    return msgs


# ---------------------------------------------------------------- benchmark
def _run_one(
    scenario_id: str,
    tool_calls: int,
    rows_per_call: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one scenario through the balanced pipeline and capture metrics."""
    msgs = _scenario(tool_calls=tool_calls, rows_per_call=rows_per_call, seed=seed)
    items_before = lc.to_items(msgs)
    tokens_before = count_items_tokens(items_before, model="gpt-4o-mini")

    pipe = presets.balanced(model="gpt-4o-mini", max_tokens=6000)
    t0 = time.perf_counter()
    res = pipe.run(items_before, task=msgs[-1].content)
    elapsed_ms_outer = (time.perf_counter() - t0) * 1000.0

    overall = {
        "scenario_id": scenario_id,
        "tool_calls": tool_calls,
        "rows_per_call": rows_per_call,
        "seed": seed,
        "msgs_before": len(items_before),
        "msgs_after": len(res.items),
        "tokens_before": int(res.tokens_before),
        "tokens_after": int(res.tokens_after),
        "saved_tokens": int(res.saved_tokens),
        "compression_ratio": round(float(res.compression_ratio), 4),
        "elapsed_ms_pipeline_internal": round(float(res.elapsed_ms), 3),
        "elapsed_ms_outer": round(elapsed_ms_outer, 3),
        "saved_usd_gpt4o_mini": float(res.saved_usd) if res.saved_usd is not None else None,
    }
    per_layer = [{
        "scenario_id": scenario_id,
        "layer": lr.layer_name,
        "items_before": lr.items_before,
        "items_after": lr.items_after,
        "tokens_before": int(lr.tokens_before),
        "tokens_after": int(lr.tokens_after),
        "saved_tokens": int(lr.saved_tokens),
        "elapsed_ms": round(float(lr.elapsed_ms), 3),
    } for lr in res.layers]
    return overall, per_layer


def _build_grid() -> List[Tuple[str, int, int]]:
    """A fixed grid of scenarios covering the input shapes observed in production.

    Production input tokens per turn (75 measured turns): median 17{,}152,
    p95 65{,}257, max 90{,}637. The grid below brackets that range.
    """
    grid: List[Tuple[str, int, int]] = []
    for n_tools in (1, 2, 3, 5, 8):
        for rows in (10, 50, 150, 400):
            grid.append((f"tools={n_tools}_rows={rows}", n_tools, rows))
    return grid


def main(out_path: Path, layers_out_path: Path, repeats: int) -> None:
    grid = _build_grid()
    all_overall: List[Dict[str, Any]] = []
    all_layers: List[Dict[str, Any]] = []
    print(f"Running {len(grid)} scenarios x {repeats} repeats "
          f"= {len(grid) * repeats} pipeline runs ...")
    seed = 20260520
    for sid, n_tools, rows in grid:
        for r in range(repeats):
            o, ls = _run_one(sid, n_tools, rows, seed)
            all_overall.append(o)
            all_layers.extend(ls)
            seed += 1
        # Quick summary as it runs so the operator can see progress.
        last = all_overall[-repeats:]
        avg_saved = sum(x["saved_tokens"] for x in last) / repeats
        avg_ms = sum(x["elapsed_ms_pipeline_internal"] for x in last) / repeats
        avg_before = sum(x["tokens_before"] for x in last) / repeats
        avg_after = sum(x["tokens_after"] for x in last) / repeats
        print(f"  {sid:25s}  tokens {avg_before:>6.0f}->{avg_after:<6.0f}  "
              f"saved {avg_saved:>6.0f}  {avg_ms:>5.1f}ms")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_overall[0].keys()))
        w.writeheader()
        w.writerows(all_overall)
    print(f"Wrote {out_path}  ({len(all_overall)} rows)")

    layers_out_path.parent.mkdir(parents=True, exist_ok=True)
    with layers_out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_layers[0].keys()))
        w.writeheader()
        w.writerows(all_layers)
    print(f"Wrote {layers_out_path}  ({len(all_layers)} rows)")

    # ---- Console summary so the operator can sanity-check at a glance ----
    ms = [x["elapsed_ms_pipeline_internal"] for x in all_overall]
    ratios = [x["compression_ratio"] for x in all_overall]
    print("\n=== OVERHEAD (pipeline-internal wall clock, ms) ===")
    print(f"  n={len(ms)}  mean={statistics.mean(ms):.2f}  "
          f"median={statistics.median(ms):.2f}  "
          f"p95={sorted(ms)[int(0.95 * len(ms))]:.2f}  "
          f"max={max(ms):.2f}")
    print("\n=== COMPRESSION RATIO ===")
    print(f"  n={len(ratios)}  mean={statistics.mean(ratios):.3f}  "
          f"median={statistics.median(ratios):.3f}  "
          f"min={min(ratios):.3f}  max={max(ratios):.3f}")

    # Per-layer attribution summary.
    by_layer: Dict[str, List[int]] = {}
    by_layer_ms: Dict[str, List[float]] = {}
    for L in all_layers:
        by_layer.setdefault(L["layer"], []).append(L["saved_tokens"])
        by_layer_ms.setdefault(L["layer"], []).append(L["elapsed_ms"])
    print("\n=== PER-LAYER TOKEN ATTRIBUTION (saved tokens / run) ===")
    for layer, vals in by_layer.items():
        ms_vals = by_layer_ms[layer]
        print(f"  {layer:30s}  mean_saved={statistics.mean(vals):8.1f}  "
              f"sum_saved={sum(vals):>7d}  "
              f"mean_ms={statistics.mean(ms_vals):5.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/optimizer_microbench.csv", type=Path)
    ap.add_argument("--layers-out", default="/tmp/optimizer_microbench_layers.csv",
                    type=Path)
    ap.add_argument("--repeats", default=5, type=int,
                    help="repeats per scenario (default 5)")
    args = ap.parse_args()
    main(args.out, args.layers_out, args.repeats)

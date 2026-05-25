"""Generic-agent persona benchmark for ContextOptimizer.

The production case study in Section 8 of paper.tex is a single
deployment (LangGraph NL-to-SQL over BigQuery). To show that the
optimiser is shape-agnostic, this script constructs four synthetic
agent personas whose message shapes are deliberately different from
each other and from the BigQuery agent, and runs the same balanced()
preset against each. We record:

* per-call before/after/saved tokens
* per-layer attribution
* compression ratio

so that the paper can report which layers engage on which agent shape.

The personas:

  react_web_search:
      Many small tool calls (search + browse) with short, varied
      tool results. Exercises the BM25 / recency rerankers.

  code_fix:
      A handful of large file-read tool calls, with the same file
      often re-read across iterations. Exercises message dedup and
      the tool-result compression layer.

  multi_turn_research:
      A 6-10 turn research conversation where each turn pulls in
      several RAG document chunks. Conversation history grows
      monotonically. Exercises the token-budget selector and the
      sliding window.

  polling_agent:
      The same get_status tool called repeatedly until a condition
      is met. Exercises the repeated-tool-call clusterer.

All personas use deterministic fixtures (no LLM call, no network).
The optimiser is the real production code. Run inside a container
that has agent.context_optimizer + langchain_core installed::

    docker cp experiments/generic_agents.py \
        de_genai-agent-template-genai-agent-template-1:/tmp/
    docker exec -w /app \
        -e PYTHONPATH=/app \
        de_genai-agent-template-genai-agent-template-1 \
        python /tmp/generic_agents.py \
            --out /tmp/generic_agents.csv \
            --layers-out /tmp/generic_agents_layers.csv
    docker cp de_genai-agent-template-genai-agent-template-1:/tmp/generic_agents.csv \
        results/generic_agents.csv
    docker cp de_genai-agent-template-genai-agent-template-1:/tmp/generic_agents_layers.csv \
        results/generic_agents_layers.csv

The two CSVs feed the per-persona table and figure in Section 8.6 of
the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from agent.context_optimizer import presets
from agent.context_optimizer.adapters import langchain_adapter as lc
from agent.context_optimizer.core.tokens import count_items_tokens
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ----- Fixtures used by every persona

SYSTEM_PROMPT = (
    "You are a careful, tool-using assistant. Use the provided tools to "
    "answer the user's question. Cite sources when possible. Do not "
    "fabricate values."
)


def _user_query(rng: random.Random, persona: str) -> str:
    bank = {
        "react_web_search": [
            "What is the latest stable Postgres release and what were the headline changes?",
            "Find three open source vector databases and summarise their license terms.",
            "Which countries adopted the OECD pillar-two minimum tax in 2025?",
        ],
        "code_fix": [
            "The unit test test_pricing_round_trip is failing on the staging branch. Find the bug and propose a fix.",
            "Add structured logging to the order-service handler and update the matching tests.",
            "Refactor utils.config to read from environment variables instead of a YAML file.",
        ],
        "multi_turn_research": [
            "I am writing a survey on retrieval-augmented generation for code. Help me find and synthesise the most relevant 2024-2026 papers.",
            "Compare the safety positions of three major LLM providers, citing the policy docs you find.",
            "Build me a technical brief on the state of the art in agentic memory systems.",
        ],
        "polling_agent": [
            "Submit the rendering job and wait for it to finish, then download the artifact.",
            "Start the long-running migration and report back when it reaches the 'done' state.",
        ],
    }
    return rng.choice(bank[persona])


def _synth_search_result(rng: random.Random, n_hits: int) -> str:
    out = []
    for i in range(n_hits):
        out.append({
            "rank": i + 1,
            "title": f"Result_{rng.randint(1, 1000)}: a short title with some text",
            "url": f"https://example.com/{rng.randint(1, 99999)}",
            "snippet": (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
            ),
        })
    return json.dumps({"hits": out, "took_ms": rng.randint(50, 250)}, indent=2)


def _synth_file_content(rng: random.Random, lines: int) -> str:
    out = []
    for i in range(lines):
        out.append(
            f"  {i:4d}  def function_{rng.randint(0, 50)}(arg_{rng.randint(0, 5)}):"
        )
        out.append(
            f"  {i:4d}+1  return arg_{rng.randint(0, 5)} + {rng.randint(0, 99)}"
        )
    return "\n".join(out)


def _synth_rag_chunks(rng: random.Random, n_chunks: int) -> str:
    chunks = []
    para = (
        "Recent work on retrieval-augmented generation has converged on a "
        "two-step retrieve-then-read pattern in which a dense or sparse "
        "retriever surfaces a small number of candidate documents and a "
        "reader model synthesises an answer from them. Variations include "
        "iterative retrieval, multi-hop reasoning, and tool-augmented "
        "retrieval over structured data. "
    )
    for i in range(n_chunks):
        chunks.append({
            "doc_id": f"doc_{rng.randint(1, 999)}",
            "score": round(rng.random(), 3),
            "text": para * rng.randint(2, 4),
        })
    return json.dumps({"chunks": chunks}, indent=2)


def _synth_status_payload(state: str) -> str:
    """Polling payload modelled on a long-running job whose status only
    changes at terminal events. Many real APIs (long migrations, render
    farms, batch jobs) report an unchanged 'running' payload for the
    entire mid-life of the job, then a single 'done' payload at the end.
    This is the canonical case the RepeatedToolCallMerger is designed
    for: N identical responses collapsed into one with a count prefix.
    """
    return json.dumps({
        "job_id": "job_42",
        "state": state,
        "progress_pct": 50 if state == "running" else 100,
        "logs": [f"step_{i}: ok" for i in range(5)],
    }, indent=2)


# ----- Persona constructors
# Each returns a fresh LangChain message list for one synthetic turn.

def _persona_react_web_search(rng: random.Random) -> List[Any]:
    msgs: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_user_query(rng, "react_web_search")),
    ]
    for i in range(rng.randint(3, 6)):
        tc_id = f"call_{i}"
        tool = "web_search" if i % 2 == 0 else "browse_url"
        args = (
            {"query": "postgres release notes 2026"} if tool == "web_search"
            else {"url": f"https://example.com/{rng.randint(1, 99)}"}
        )
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"id": tc_id, "name": tool, "args": args}],
            )
        )
        n_hits = rng.randint(5, 10) if tool == "web_search" else 1
        msgs.append(
            ToolMessage(
                content=_synth_search_result(rng, n_hits),
                tool_call_id=tc_id,
                name=tool,
            )
        )
    return msgs


def _persona_code_fix(rng: random.Random) -> List[Any]:
    """The same file is re-read multiple times - exercises dedup."""
    msgs: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_user_query(rng, "code_fix")),
    ]
    file_content_a = _synth_file_content(rng, 80)
    file_content_b = _synth_file_content(rng, 60)
    reads = [
        ("read_file", {"path": "src/pricing.py"}, file_content_a),
        ("read_file", {"path": "tests/test_pricing.py"}, file_content_b),
        ("read_file", {"path": "src/pricing.py"}, file_content_a),  # dup
        ("read_file", {"path": "src/pricing.py"}, file_content_a),  # dup
        ("run_tests", {"path": "tests/test_pricing.py"},
         "FAILED tests/test_pricing.py::test_round_trip"),
    ]
    for i, (tool, args, result) in enumerate(reads):
        tc_id = f"call_{i}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"id": tc_id, "name": tool, "args": args}],
            )
        )
        msgs.append(
            ToolMessage(content=result, tool_call_id=tc_id, name=tool)
        )
    return msgs


def _persona_multi_turn_research(rng: random.Random) -> List[Any]:
    """6-10 turns of RAG, history grows monotonically."""
    msgs: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_user_query(rng, "multi_turn_research")),
    ]
    n_turns = rng.randint(6, 10)
    for turn in range(n_turns):
        tc_id = f"rag_{turn}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "id": tc_id,
                    "name": "retrieve_documents",
                    "args": {"query": f"sub-question {turn}", "k": 8},
                }],
            )
        )
        msgs.append(
            ToolMessage(
                content=_synth_rag_chunks(rng, n_chunks=rng.randint(6, 12)),
                tool_call_id=tc_id,
                name="retrieve_documents",
            )
        )
        msgs.append(
            AIMessage(content=(
                f"Synthesis of turn {turn}: based on the retrieved "
                f"chunks, I would summarise the key points as follows. "
                f"First, the literature converges on retrieve-then-read. "
                f"Second, iterative retrieval helps on multi-hop. Third, "
                f"tool-augmented variants are gaining ground."
            ))
        )
        msgs.append(HumanMessage(content=f"Follow-up question {turn}, please dig deeper."))
    return msgs


def _persona_polling_agent(rng: random.Random) -> List[Any]:
    """The same get_status tool called 5-12 times - exercises clustering."""
    msgs: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_user_query(rng, "polling_agent")),
    ]
    n_polls = rng.randint(5, 12)
    for i in range(n_polls):
        tc_id = f"poll_{i}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{
                    "id": tc_id,
                    "name": "get_status",
                    "args": {"job_id": "job_42"},
                }],
            )
        )
        state = "running" if i < n_polls - 1 else "done"
        msgs.append(
            ToolMessage(
                content=_synth_status_payload(state),
                tool_call_id=tc_id,
                name="get_status",
            )
        )
    return msgs


PERSONAS: Dict[str, Callable[[random.Random], List[Any]]] = {
    "react_web_search": _persona_react_web_search,
    "code_fix": _persona_code_fix,
    "multi_turn_research": _persona_multi_turn_research,
    "polling_agent": _persona_polling_agent,
}


# ----- The benchmark loop

def _run_one(persona: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    msgs = PERSONAS[persona](rng)
    user_query = next(
        (m.content for m in msgs if isinstance(m, HumanMessage)),
        "",
    )

    items_before = lc.to_items(msgs)
    pipeline = presets.balanced()
    result = pipeline.run(items_before, task=user_query)

    layer_rows: List[Dict[str, Any]] = []
    for lr in result.layers:
        layer_rows.append({
            "persona": persona,
            "seed": seed,
            "layer": lr.layer_name,
            "items_before": lr.items_before,
            "items_after": lr.items_after,
            "tokens_before": lr.tokens_before,
            "tokens_after": lr.tokens_after,
            "saved_tokens": lr.saved_tokens,
            "elapsed_ms": round(lr.elapsed_ms, 3),
        })

    return {
        "summary": {
            "persona": persona,
            "seed": seed,
            "messages_in_input": len(msgs),
            "tool_calls_in_input": sum(
                1 for m in msgs
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            ),
            "tool_results_in_input": sum(
                1 for m in msgs if isinstance(m, ToolMessage)
            ),
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "saved_tokens": result.saved_tokens,
            "compression_ratio": round(result.compression_ratio, 4),
            "elapsed_ms_total": round(result.elapsed_ms, 3),
        },
        "layers": layer_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="generic_agents.csv")
    p.add_argument("--layers-out", default="generic_agents_layers.csv")
    p.add_argument("--repeats", type=int, default=20,
                   help="Random seeds per persona.")
    p.add_argument("--seed-base", type=int, default=20260520)
    args = p.parse_args()

    summary_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []

    for persona in PERSONAS:
        for r in range(args.repeats):
            res = _run_one(persona, seed=args.seed_base + r)
            summary_rows.append(res["summary"])
            layer_rows.extend(res["layers"])

    # Write summary CSV
    out_path = Path(args.out)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {out_path} ({len(summary_rows)} rows)")

    layers_path = Path(args.layers_out)
    with layers_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(layer_rows[0].keys()))
        w.writeheader()
        w.writerows(layer_rows)
    print(f"Wrote {layers_path} ({len(layer_rows)} rows)")

    # Console summary so the operator can sanity-check at a glance.
    print()
    print(f"{'persona':22s}  {'runs':>4s}  {'tok_before':>10s}  "
          f"{'tok_after':>10s}  {'ratio':>6s}  {'ms':>6s}")
    print("-" * 70)
    for persona in PERSONAS:
        rows = [r for r in summary_rows if r["persona"] == persona]
        avg_before = sum(r["tokens_before"] for r in rows) / len(rows)
        avg_after = sum(r["tokens_after"] for r in rows) / len(rows)
        avg_ratio = avg_after / avg_before if avg_before else 1.0
        avg_ms = sum(r["elapsed_ms_total"] for r in rows) / len(rows)
        print(f"{persona:22s}  {len(rows):>4d}  {avg_before:>10.0f}  "
              f"{avg_after:>10.0f}  {avg_ratio:>6.3f}  {avg_ms:>6.1f}")

    print()
    print("Per-layer engagement (mean saved tokens per run, across all personas):")
    by_layer: Dict[str, List[int]] = {}
    for r in layer_rows:
        by_layer.setdefault(r["layer"], []).append(r["saved_tokens"])
    for layer, saved_list in sorted(by_layer.items()):
        mean = sum(saved_list) / len(saved_list)
        engaged = sum(1 for s in saved_list if s > 0)
        print(f"  {layer:40s}  mean_saved={mean:>7.0f}  engaged={engaged:>3d}/{len(saved_list)}")


if __name__ == "__main__":
    main()

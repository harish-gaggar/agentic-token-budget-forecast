"""Training-free truncation baseline (B9) vs ContextOptimizer persona CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, List

# Fixture helpers mirrored from generic_agents.py (stdlib only).
SYSTEM_PROMPT = (
    "You are a careful, tool-using assistant. Use the provided tools to "
    "answer the user's question."
)


def _user_query(rng: random.Random, persona: str) -> str:
    bank = {
        "react_web_search": ["What is the latest stable Postgres release?"],
        "code_fix": ["Fix the failing unit test on staging."],
        "multi_turn_research": ["Survey RAG for code, 2024-2026."],
        "polling_agent": ["Submit the job and wait until done."],
    }
    return rng.choice(bank[persona])


def _synth_search_result(rng: random.Random, n_hits: int) -> str:
    hits = [
        {
            "rank": i + 1,
            "title": f"Result_{rng.randint(1, 1000)}",
            "snippet": "Lorem ipsum " * 20,
        }
        for i in range(n_hits)
    ]
    return json.dumps({"hits": hits}, indent=2)


def _synth_file_content(rng: random.Random, lines: int) -> str:
    return "\n".join(f"  {i:4d}  line_{rng.randint(0, 99)}" for i in range(lines))


def _synth_rag_chunks(rng: random.Random, n_chunks: int) -> str:
    para = "RAG survey paragraph. " * 40
    return json.dumps(
        {"chunks": [{"text": para * rng.randint(2, 4)} for _ in range(n_chunks)]},
        indent=2,
    )


def _synth_status_payload(state: str) -> str:
    return json.dumps({"job_id": "job_42", "state": state}, indent=2)


def _persona_react(rng: random.Random) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_query(rng, "react_web_search")},
    ]
    for i in range(rng.randint(3, 6)):
        tool = "web_search" if i % 2 == 0 else "browse_url"
        n = rng.randint(5, 10) if tool == "web_search" else 1
        msgs.append({"role": "assistant", "content": ""})
        msgs.append({"role": "tool", "content": _synth_search_result(rng, n)})
    return msgs


def _persona_code_fix(rng: random.Random) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_query(rng, "code_fix")},
    ]
    fa, fb = _synth_file_content(rng, 80), _synth_file_content(rng, 60)
    for content in (fa, fb, fa, fa, "FAILED test_round_trip"):
        msgs.append({"role": "assistant", "content": ""})
        msgs.append({"role": "tool", "content": content})
    return msgs


def _persona_research(rng: random.Random) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_query(rng, "multi_turn_research")},
    ]
    for turn in range(rng.randint(6, 10)):
        msgs.append({"role": "assistant", "content": ""})
        msgs.append(
            {
                "role": "tool",
                "content": _synth_rag_chunks(rng, rng.randint(6, 12)),
            }
        )
        msgs.append({"role": "assistant", "content": f"Synthesis turn {turn}." * 30})
        msgs.append({"role": "user", "content": f"Follow-up {turn}."})
    return msgs


def _persona_polling(rng: random.Random) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_query(rng, "polling_agent")},
    ]
    n = rng.randint(5, 12)
    for i in range(n):
        msgs.append({"role": "assistant", "content": ""})
        msgs.append(
            {
                "role": "tool",
                "content": _synth_status_payload("running" if i < n - 1 else "done"),
            }
        )
    return msgs


PERSONAS: Dict[str, Callable[[random.Random], List[Dict[str, str]]]] = {
    "react_web_search": _persona_react,
    "code_fix": _persona_code_fix,
    "multi_turn_research": _persona_research,
    "polling_agent": _persona_polling,
}

MAX_TOKENS = 8000
TOOL_CHAR_CAP = 1200  # aggressive head-tail cap per tool message


def _est_tokens(text: str) -> int:
    # Slightly pessimistic vs.\ char/4 so budgets bite on large tool JSON.
    return max(1, int(len(text) / 3.2))


def truncate_baseline(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    trimmed = []
    for m in msgs:
        c = m["content"]
        if m["role"] == "tool":
            if len(c) > TOOL_CHAR_CAP:
                h = TOOL_CHAR_CAP // 2
                c = c[:h] + "\n…[truncated]…\n" + c[-h:]
        elif m["role"] == "assistant" and len(c) > 2000:
            c = c[:1000] + "\n…\n" + c[-1000:]
        trimmed.append({"role": m["role"], "content": c})

    def total(ms: List[Dict[str, str]]) -> int:
        return sum(_est_tokens(m["content"]) for m in ms)

    while total(trimmed) > MAX_TOKENS and len(trimmed) > 3:
        for i, m in enumerate(trimmed):
            if m["role"] not in ("system", "user"):
                trimmed.pop(i)
                break
        else:
            break
    return trimmed


def _resolve_out(path: Path) -> Path:
    """--out may be a directory (Makefile) or a CSV file path."""
    if path.suffix.lower() == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / "compression_baseline.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results", type=Path)
    p.add_argument("--ctx-opt", default="results/generic_agents.csv", type=Path)
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--seed-base", type=int, default=20260520)
    args = p.parse_args()

    rows = []
    for persona, builder in PERSONAS.items():
        for r in range(args.repeats):
            seed = args.seed_base + r
            msgs = builder(random.Random(seed))
            before = sum(_est_tokens(m["content"]) for m in msgs)
            after = sum(_est_tokens(m["content"]) for m in truncate_baseline(msgs))
            rows.append({
                "persona": persona,
                "seed": seed,
                "tokens_before": before,
                "tokens_after": after,
                "reduction_pct": round(100 * (before - after) / before, 2) if before else 0,
            })

    out = _resolve_out(Path(args.out))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ctx: Dict[str, List[float]] = {}
    ctx_opt = Path(args.ctx_opt)
    if ctx_opt.exists():
        with ctx_opt.open() as fh:
            for row in csv.DictReader(fh):
                ctx.setdefault(row["persona"], []).append(
                    100 * (1 - float(row["compression_ratio"]))
                )

    summary = []
    for persona in PERSONAS:
        pr = [r for r in rows if r["persona"] == persona]
        t_mean = sum(r["reduction_pct"] for r in pr) / len(pr)
        c_list = ctx.get(persona, [])
        c_mean = sum(c_list) / len(c_list) if c_list else 0.0
        summary.append({
            "persona": persona,
            "truncate_reduction_pct": round(t_mean, 1),
            "ctxopt_reduction_pct": round(c_mean, 1),
        })

    sum_path = out.with_name("compression_baseline_summary.csv")
    with sum_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {out} and {sum_path}")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()

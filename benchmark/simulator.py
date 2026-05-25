"""Ground-truth token consumption simulator.

We don't have permission to ship live SWE-Bench / AppWorld execution
traces, so the "ground truth" for the benchmark is produced by a stochastic
simulator calibrated on per-stage means reported in Qiu et al. (Tokenomics,
2025) and Tokalator (Ulan uulu et al., 2026). The simulator's per-task-type
means deliberately *differ* from TABF's heuristic lookup so that no model
can trivially "win" by memorising the table; it also injects heteroscedastic
noise on the output stage for documentation and research queries, which is
where the literature reports the largest variance.

The simulator is fully seeded. Given the same seed and corpus, it always
produces identical traces.
"""

from __future__ import annotations
import hashlib
from typing import Dict, Tuple, List
import numpy as np


def _det_seed(seed: int, key: str) -> int:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


# Per-stage *means* used by the simulator at MEDIUM complexity.
# These were chosen to be close to but not identical to TABF's heuristic
# table, with a couple of types shifted by 10-25% to model real-world
# calibration drift.
SIM_BASE: Dict[str, Tuple[int, int, int, int, int, int]] = {
    #                 sys  tools  mem   ctx   reason  out
    "code_generation":  (820, 540,  480, 1100, 2050, 1100),
    "bug_fix":          (760, 470,  720, 2200, 2400,  720),
    "refactoring":      (860, 520,  900, 2350, 2700, 1620),
    "documentation":    (560, 220,  280, 1700,  720, 2300),
    "multi_step_plan":  (920, 760,  820, 1700, 3100, 1150),
    "data_analysis":    (680, 660,  540, 3250, 1850, 1620),
    "research_query":   (640,1100,  440, 3800, 1400, 1300),
    "tool_use_chain":   (820,1620,  640, 2150, 1850,  760),
}

SIM_CMULT = {"low": 0.62, "medium": 1.0, "high": 1.55, "expert": 2.30}

# Noise on each stage: log-normal sigma. Output stage is wider for docs/research.
_DEFAULT_SIGMA = {
    "system": 0.04, "tools": 0.07, "memory": 0.10,
    "context": 0.12, "reasoning": 0.14, "output": 0.12,
}
_OUTPUT_SIGMA_BY_TYPE = {
    "documentation": 0.32,
    "research_query": 0.30,
    "multi_step_plan": 0.20,
}

# Probability that the simulator promotes complexity one notch ("scope creep")
_SCOPE_CREEP_P = 0.08
_LEVELS = ["low", "medium", "high", "expert"]


def _promote(level: str) -> str:
    i = _LEVELS.index(level)
    return _LEVELS[min(i + 1, len(_LEVELS) - 1)]


def _word_count(task: str) -> int:
    return len(task.split())


def simulate_trace(
    task: str,
    latent_type: str,
    latent_complexity: str,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """Return a dict with one integer per stage plus 'total' and 'output'.

    Beyond the base[type] * cmult[complexity] structure that TABF's
    heuristic models explicitly, the simulator injects *signal-dependent
    residuals* the heuristic does not capture (e.g. question count boosts
    reasoning tokens, entity count boosts tool/context tokens). These
    residuals are what the GBT regressor is meant to learn.
    """
    cx = latent_complexity
    if rng.random() < _SCOPE_CREEP_P:
        cx = _promote(cx)

    mult = SIM_CMULT[cx]
    base = SIM_BASE[latent_type]
    stages = ["system", "tools", "memory", "context", "reasoning", "output"]
    means = dict(zip(stages, base))

    # Surface signals the simulator uses for hidden residual effects.
    low = task.lower()
    wc = _word_count(task)
    q_count = task.count("?")
    ent_count = sum(1 for tok in task.split()
                    if tok[:1].isupper() and tok != tok.upper())
    has_compare = any(w in low for w in ("compare", "versus", " vs ", "tradeoff"))

    # 1) Long descriptions retrieve more context (the heuristic captures this).
    means["context"] *= 1.0 + max(0, wc - 20) * 0.012
    # 2) Questions strongly inflate reasoning (heuristic does NOT model this).
    means["reasoning"] *= 1.0 + 0.32 * q_count
    means["output"]    *= 1.0 + 0.10 * q_count
    # 3) Entity-rich tasks pull more retrieved context AND tools.
    means["context"] *= 1.0 + 0.07 * ent_count
    means["tools"]   *= 1.0 + 0.08 * ent_count
    # 4) Explicit comparisons inflate reasoning + output.
    if has_compare:
        means["reasoning"] *= 1.40
        means["output"]    *= 1.30
    # 5) Multi-step phrasing inflates memory and reasoning.
    if any(w in low for w in ("then", "next", "finally", "step")):
        means["memory"]    *= 1.45
        means["reasoning"] *= 1.15
    # 6) File references increase context retrieval beyond the type baseline.
    if any(w in low for w in ("csv", "jsonl", "parquet", "tsv", "json")):
        means["context"] *= 1.25
    # 7) Tasks that touch deployment/migration drag in more tool-schema tokens.
    if any(w in low for w in ("deploy", "migration", "rollback", "runbook")):
        means["tools"] *= 1.35

    out_sigma = _OUTPUT_SIGMA_BY_TYPE.get(latent_type, _DEFAULT_SIGMA["output"])
    sigmas = dict(_DEFAULT_SIGMA, output=out_sigma)

    trace = {}
    for st in stages:
        mu = float(means[st]) * mult
        sigma = sigmas[st]
        mu_ln = np.log(max(mu, 1.0)) - 0.5 * sigma * sigma
        val = float(np.exp(rng.normal(mu_ln, sigma)))
        trace[st] = int(max(round(val), 1))

    trace["input_tokens"] = sum(trace[s] for s in stages if s != "output")
    trace["total"] = trace["input_tokens"] + trace["output"]
    trace["latent_complexity_realised"] = cx
    return trace


def ground_truth_for_task(task: dict, seed: int) -> Dict[str, int]:
    """Convenience wrapper that builds a per-task RNG so each row is
    independently reproducible from (corpus_seed, task_id)."""
    rng = np.random.default_rng(_det_seed(seed, task["id"]))
    return simulate_trace(
        task["text"],
        task["latent_type"],
        task["latent_complexity_hint"],
        rng,
    )


def simulate_dataset(corpus: List[dict], seed: int) -> List[dict]:
    rows = []
    for task in corpus:
        gt = ground_truth_for_task(task, seed)
        rows.append({**task, "gt": gt})
    return rows

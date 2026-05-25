"""Heuristic per-stage budget lookup, scaled by complexity."""

from .forecast import BudgetForecast
from .extractor import Signals


# Base allocations at MEDIUM complexity, per task type.
# Columns: (system, tools, memory, context, reasoning, output)
BASE_TABLE = {
    "code_generation":  ( 800,  600,  400, 1200, 1800, 1200),
    "bug_fix":          ( 800,  500,  600, 2000, 2200,  800),
    "refactoring":      ( 800,  500,  800, 2500, 2500, 1500),
    "documentation":    ( 600,  200,  300, 1500,  800, 2000),
    "multi_step_plan":  ( 900,  800,  700, 1500, 2800, 1200),
    "data_analysis":    ( 700,  700,  500, 3000, 2000, 1500),
    "research_query":   ( 600, 1200,  400, 3500, 1500, 1000),
    "tool_use_chain":   ( 800, 1500,  600, 2000, 2000,  800),
}

COMPLEXITY_MULT = {"low": 0.65, "medium": 1.0, "high": 1.6, "expert": 2.4}


class HeuristicBudgetModel:
    """Deterministic lookup-table budget model. No training required."""

    def __init__(
        self,
        base_table=None,
        complexity_mult=None,
        word_adj_per_word: float = 0.008,
        word_adj_threshold: int = 20,
    ):
        self.base_table = base_table if base_table is not None else BASE_TABLE
        self.cm = complexity_mult if complexity_mult is not None else COMPLEXITY_MULT
        self.word_adj_per_word = word_adj_per_word
        self.word_adj_threshold = word_adj_threshold

    def forecast(self, task_type: str, complexity: str, signals: Signals) -> BudgetForecast:
        s, t, m, c, r, o = self.base_table[task_type]
        mult = self.cm[complexity]
        w_adj = 1.0 + max(0, signals.word_count - self.word_adj_threshold) * self.word_adj_per_word
        return BudgetForecast(
            system=int(round(s * mult)),
            tools=int(round(t * mult)),
            memory=int(round(m * mult)),
            context=int(round(c * mult * w_adj)),
            reasoning=int(round(r * mult)),
            output=int(round(o * mult)),
            task_type=task_type,
            complexity=complexity,
            confidence=0.79,
            method="heuristic",
        )

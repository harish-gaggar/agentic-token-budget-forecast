from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class BudgetForecast:
    """A per-stage token-budget forecast for one agentic task."""

    system: int
    tools: int
    memory: int
    context: int
    reasoning: int
    output: int
    task_type: str
    complexity: str
    confidence: float
    method: str  # "heuristic" or "ml_gbt"

    @property
    def input_tokens(self) -> int:
        return self.system + self.tools + self.memory + self.context + self.reasoning

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["input_tokens"] = self.input_tokens
        d["total_tokens"] = self.total_tokens
        return d

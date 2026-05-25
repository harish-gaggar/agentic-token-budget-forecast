"""High-level TABF facade that chains signal extraction, classification,
and budget prediction."""

from typing import Optional

from .forecast import BudgetForecast
from .extractor import extract_signals
from .classifier import classify
from .heuristic import HeuristicBudgetModel
from .ml import GBTBudgetModel


class TABF:
    def __init__(self, ml_model: Optional[GBTBudgetModel] = None,
                 heuristic: Optional[HeuristicBudgetModel] = None):
        self.heuristic = heuristic or HeuristicBudgetModel()
        self.ml = ml_model

    def forecast(self, task: str) -> BudgetForecast:
        sig = extract_signals(task)
        ktype, cx = classify(task, sig)
        if self.ml is not None:
            return self.ml.forecast(ktype, cx, sig)
        return self.heuristic.forecast(ktype, cx, sig)

"""TABF: Task-Aware Budget Forecaster.

Public API:
    >>> from tabf import TABF
    >>> tabf = TABF()
    >>> fc = tabf.forecast("Refactor the user_auth module to use JWT tokens.")
    >>> print(fc.total_tokens, fc.task_type, fc.complexity)
"""

from .forecast import BudgetForecast
from .extractor import extract_signals, Signals
from .classifier import classify, TASK_TYPES, COMPLEXITY_LEVELS
from .heuristic import HeuristicBudgetModel
from .ml import GBTBudgetModel
from .api import TABF

__all__ = [
    "TABF",
    "BudgetForecast",
    "Signals",
    "extract_signals",
    "classify",
    "TASK_TYPES",
    "COMPLEXITY_LEVELS",
    "HeuristicBudgetModel",
    "GBTBudgetModel",
]

__version__ = "0.1.0"

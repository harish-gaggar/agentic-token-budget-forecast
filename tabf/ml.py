"""Optional GBT regressor that refines heuristic forecasts when traces exist."""

from typing import Iterable, Tuple, List
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .forecast import BudgetForecast
from .extractor import Signals
from .heuristic import HeuristicBudgetModel
from .classifier import TASK_TYPES, COMPLEXITY_LEVELS


def _feature_vector(sig: Signals, task_type: str, complexity: str) -> List[float]:
    return sig.as_vector() + [
        TASK_TYPES.index(task_type),
        COMPLEXITY_LEVELS.index(complexity),
    ]


class GBTBudgetModel:
    """Two GBT regressors: one for total input tokens, one for output tokens.

    Per-stage allocation reuses the heuristic ratios so the predicted total is
    distributed in a way consistent with the trace data the heuristic was
    calibrated on.
    """

    def __init__(self, heuristic: HeuristicBudgetModel = None,
                 n_estimators: int = 120, max_depth: int = 4,
                 learning_rate: float = 0.08, random_state: int = 0):
        self.heuristic = heuristic or HeuristicBudgetModel()
        self._kw = dict(n_estimators=n_estimators, max_depth=max_depth,
                        learning_rate=learning_rate, random_state=random_state)
        self.m_input = GradientBoostingRegressor(**self._kw)
        self.m_output = GradientBoostingRegressor(**self._kw)
        self._fitted = False

    def fit(self, traces: Iterable[Tuple[Signals, str, str, int, int]]):
        """traces: iterable of (signals, task_type, complexity, true_input, true_output)."""
        X, y_in, y_out = [], [], []
        for sig, ktype, cx, true_in, true_out in traces:
            X.append(_feature_vector(sig, ktype, cx))
            y_in.append(true_in)
            y_out.append(true_out)
        X = np.asarray(X, dtype=float)
        self.m_input.fit(X, np.asarray(y_in, dtype=float))
        self.m_output.fit(X, np.asarray(y_out, dtype=float))
        self._fitted = True
        return self

    def forecast(self, task_type: str, complexity: str, signals: Signals) -> BudgetForecast:
        if not self._fitted:
            raise RuntimeError("GBTBudgetModel must be fit before forecasting.")
        base = self.heuristic.forecast(task_type, complexity, signals)

        x = np.asarray([_feature_vector(signals, task_type, complexity)], dtype=float)
        pred_in = max(int(self.m_input.predict(x)[0]), 1)
        pred_out = max(int(self.m_output.predict(x)[0]), 1)

        # Redistribute predicted input across stages using heuristic ratios.
        heur_in = base.input_tokens or 1
        ratios = {
            "system":    base.system    / heur_in,
            "tools":     base.tools     / heur_in,
            "memory":    base.memory    / heur_in,
            "context":   base.context   / heur_in,
            "reasoning": base.reasoning / heur_in,
        }
        return BudgetForecast(
            system=int(round(pred_in * ratios["system"])),
            tools=int(round(pred_in * ratios["tools"])),
            memory=int(round(pred_in * ratios["memory"])),
            context=int(round(pred_in * ratios["context"])),
            reasoning=int(round(pred_in * ratios["reasoning"])),
            output=pred_out,
            task_type=task_type,
            complexity=complexity,
            confidence=0.91,
            method="ml_gbt",
        )

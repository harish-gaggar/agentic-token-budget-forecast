"""GBT regressors fit directly to each stage (no heuristic redistribution)."""

from typing import Iterable, List, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .forecast import BudgetForecast
from .extractor import Signals
from .heuristic import HeuristicBudgetModel
from .classifier import TASK_TYPES, COMPLEXITY_LEVELS
from .ml import _feature_vector


STAGES = ("system", "tools", "memory", "context", "reasoning", "output")


class GBTBudgetModelStagewise:
    """Six independent GBT regressors, one per stage."""

    def __init__(self, heuristic: HeuristicBudgetModel = None,
                 n_estimators: int = 120, max_depth: int = 4,
                 learning_rate: float = 0.08, random_state: int = 0):
        self.heuristic = heuristic or HeuristicBudgetModel()
        kw = dict(n_estimators=n_estimators, max_depth=max_depth,
                  learning_rate=learning_rate, random_state=random_state)
        self._models = {s: GradientBoostingRegressor(**kw) for s in STAGES}
        self._fitted = False

    def fit(self, traces: Iterable[Tuple[Signals, str, str, dict]]):
        """traces: (signals, task_type, complexity, gt_stage_dict)."""
        X_by = {s: [] for s in STAGES}
        y_by = {s: [] for s in STAGES}
        for sig, ktype, cx, gt in traces:
            x = _feature_vector(sig, ktype, cx)
            for s in STAGES:
                X_by[s].append(x)
                y_by[s].append(float(gt[s]))
        for s in STAGES:
            self._models[s].fit(np.asarray(X_by[s], dtype=float),
                                np.asarray(y_by[s], dtype=float))
        self._fitted = True
        return self

    def forecast(self, task_type: str, complexity: str, signals: Signals) -> BudgetForecast:
        if not self._fitted:
            raise RuntimeError("GBTBudgetModelStagewise must be fit before forecasting.")
        x = np.asarray([_feature_vector(signals, task_type, complexity)], dtype=float)
        vals = {s: max(int(self._models[s].predict(x)[0]), 1) for s in STAGES}
        return BudgetForecast(
            system=vals["system"], tools=vals["tools"], memory=vals["memory"],
            context=vals["context"], reasoning=vals["reasoning"], output=vals["output"],
            task_type=task_type, complexity=complexity,
            confidence=0.91, method="ml_gbt_stagewise",
        )

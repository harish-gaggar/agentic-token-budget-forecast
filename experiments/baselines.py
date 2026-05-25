"""Baselines used for comparison against TABF.

The three original baselines (Uniform, TypeOnly, LengthProportional) are
naive on purpose: they establish the floor that any task-aware predictor
has to clear. The remaining baselines are learned regressors that fit on
the same feature space TABF uses, plus a TF-IDF bag-of-words variant
that lets us check whether richer lexical features close the gap to the
hand-engineered signals.

All learned baselines use scikit-learn only; no LightGBM/XGBoost
dependency so the repo stays installable from a clean Python.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from tabf.extractor import extract_signals, Signals
from tabf.classifier import classify, TASK_TYPES, COMPLEXITY_LEVELS
from tabf.heuristic import BASE_TABLE


@dataclass
class BaselinePrediction:
    total: int
    method: str


# -----------------------------------------------------------------------------
# Naive baselines (B1-B3): no learned features.
# -----------------------------------------------------------------------------

class UniformBaseline:
    """B1. Predict the corpus-mean total for every task."""

    name = "B1_uniform"

    def fit(self, totals: List[int]):
        self.mean = int(round(float(np.mean(totals))))
        return self

    def predict(self, task: str) -> BaselinePrediction:
        return BaselinePrediction(total=self.mean, method=self.name)


class TypeOnlyHeuristic:
    """B2. Look up base allocation per task type, no complexity scaling."""

    name = "B2_type_only"

    def predict(self, task: str) -> BaselinePrediction:
        sig = extract_signals(task)
        ktype, _ = classify(task, sig)
        base = BASE_TABLE[ktype]
        return BaselinePrediction(total=int(sum(base)), method=self.name)


class LengthProportional:
    """B3. Scale a corpus-average linearly with task word count."""

    name = "B3_length"

    def fit(self, tasks_words: List[int], totals: List[int]):
        words = np.asarray(tasks_words, dtype=float)
        totals = np.asarray(totals, dtype=float)
        self.slope = float(np.sum(words * totals) / max(np.sum(words * words), 1.0))
        return self

    def predict(self, task: str) -> BaselinePrediction:
        wc = len(task.split())
        return BaselinePrediction(total=int(max(round(self.slope * wc), 1)),
                                  method=self.name)


# -----------------------------------------------------------------------------
# Learned baselines on the same hand-engineered feature set TABF uses.
# Fit on (signals, task_type, complexity) -> total tokens. This is a
# strict apples-to-apples comparison with TABF_gbt: same features,
# different regressor families.
# -----------------------------------------------------------------------------

def _hand_features(task: str) -> List[float]:
    sig = extract_signals(task)
    ktype, cx = classify(task, sig)
    return sig.as_vector() + [
        TASK_TYPES.index(ktype),
        COMPLEXITY_LEVELS.index(cx),
    ]


class _HandFeatureRegressor:
    """Base wrapper: fit a sklearn regressor on the hand-engineered features."""

    name = "override_me"

    def __init__(self, estimator):
        self._est = estimator

    def fit(self, tasks: List[str], totals: List[int]):
        X = np.asarray([_hand_features(t) for t in tasks], dtype=float)
        y = np.asarray(totals, dtype=float)
        self._est.fit(X, y)
        return self

    def predict(self, task: str) -> BaselinePrediction:
        x = np.asarray([_hand_features(task)], dtype=float)
        pred = max(int(round(float(self._est.predict(x)[0]))), 1)
        return BaselinePrediction(total=pred, method=self.name)


class RidgeOnFeatures(_HandFeatureRegressor):
    """B4. Ridge regression on TABF's 10-dim hand-engineered features."""

    name = "B4_ridge_hand"

    def __init__(self, alpha: float = 1.0, random_state: int = 0):
        super().__init__(Ridge(alpha=alpha, random_state=random_state))


class RandomForestOnFeatures(_HandFeatureRegressor):
    """B5. RandomForest on TABF's 10-dim hand-engineered features."""

    name = "B5_rf_hand"

    def __init__(self, n_estimators: int = 200, max_depth: int = 8,
                 random_state: int = 0):
        super().__init__(RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=1,
        ))


class HistGBOnFeatures(_HandFeatureRegressor):
    """B6. HistGradientBoosting on TABF's 10-dim hand-engineered features.

    This is the sklearn analogue of LightGBM that the reviewer asked
    for. We use it instead of LightGBM specifically to keep the
    install-from-clean-Python story for reviewers.
    """

    name = "B6_histgb_hand"

    def __init__(self, max_iter: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.06, random_state: int = 0):
        super().__init__(HistGradientBoostingRegressor(
            max_iter=max_iter, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
        ))


# -----------------------------------------------------------------------------
# Learned baselines on a richer TF-IDF bag-of-words representation.
# Same regressor family (Ridge, HistGB) so the only thing that changes
# is the feature set. This tells us whether hand-engineered signals or
# raw lexical features carry the predictive power.
# -----------------------------------------------------------------------------

class _TfidfRegressor:
    """Base wrapper: TfidfVectorizer + sklearn regressor on raw text."""

    name = "override_me"

    def __init__(self, estimator, max_features: int = 2000):
        self._vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
        )
        self._est = estimator

    def fit(self, tasks: List[str], totals: List[int]):
        X = self._vec.fit_transform(tasks)
        y = np.asarray(totals, dtype=float)
        self._est.fit(X, y)
        return self

    def predict(self, task: str) -> BaselinePrediction:
        X = self._vec.transform([task])
        pred = max(int(round(float(self._est.predict(X)[0]))), 1)
        return BaselinePrediction(total=pred, method=self.name)


class RidgeOnTfidf(_TfidfRegressor):
    """B7. Ridge over TF-IDF 1-2grams of the raw task description.

    No task-type, no complexity, no hand signals. Tests whether the
    lexical surface alone is enough.
    """

    name = "B7_ridge_tfidf"

    def __init__(self, alpha: float = 1.0):
        super().__init__(Ridge(alpha=alpha, random_state=0))


class HistGBOnTfidf(_TfidfRegressor):
    """B8. HistGradientBoosting over TF-IDF 1-2grams of the raw task."""

    name = "B8_histgb_tfidf"

    def __init__(self, max_iter: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.06):
        # HistGB doesn't accept sparse input; wrap to densify.
        super().__init__(HistGradientBoostingRegressor(
            max_iter=max_iter, max_depth=max_depth,
            learning_rate=learning_rate, random_state=0,
        ))

    def fit(self, tasks: List[str], totals: List[int]):
        X = self._vec.fit_transform(tasks).toarray()
        y = np.asarray(totals, dtype=float)
        self._est.fit(X, y)
        return self

    def predict(self, task: str) -> BaselinePrediction:
        X = self._vec.transform([task]).toarray()
        pred = max(int(round(float(self._est.predict(X)[0]))), 1)
        return BaselinePrediction(total=pred, method=self.name)

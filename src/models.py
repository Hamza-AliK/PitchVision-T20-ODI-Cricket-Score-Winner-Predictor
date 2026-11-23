from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BasePredictor(ABC):
    """Abstract base class. X must be in chronological order (oldest first)."""
    VALIDATION_FRACTION = 0.15

    def __init__(self, name):
        self._model = None
        self._is_fitted = False
        self._name = name
        self.selected = None          # name of the candidate chosen on validation data
        self.validation = {}          # candidate name -> validation score

    @property
    def name(self):
        return self._name

    def _require_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(f"{self._name} has not been fitted yet.")

    def _split(self, X):
        return int(len(X) * (1 - self.VALIDATION_FRACTION))

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass


class ScorePredictor(BasePredictor):
    """Regression: Ridge vs gradient boosting, chosen on a later-in-time validation slice."""

    def __init__(self):
        super().__init__(name="Score Predictor (Regression)")
        self._lo = self._hi = None
        self._train_mean = None

    @staticmethod
    def _candidates():
        return {
            "Ridge Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
            "Gradient Boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                                              l2_regularization=1.0, random_state=0),
        }

    def fit(self, X, y):
        cut = self._split(X)
        self.validation = {}
        for name, est in self._candidates().items():
            est.fit(X.iloc[:cut], y.iloc[:cut])
            self.validation[name] = mean_absolute_error(y.iloc[cut:], est.predict(X.iloc[cut:]))
        self.selected = min(self.validation, key=self.validation.get)
        self._model = self._candidates()[self.selected].fit(X, y)        # refit on all training data

        # 10th / 90th percentile models give a score range, not just one number
        def q(alpha):
            return HistGradientBoostingRegressor(loss="quantile", quantile=alpha, max_depth=3,
                                                 learning_rate=0.05, max_iter=150, random_state=0).fit(X, y)
        self._lo, self._hi = q(0.1), q(0.9)
        self._train_mean = float(y.mean())
        self._is_fitted = True
        return self

    def predict(self, X):
        self._require_fitted()
        return self._model.predict(X)

    def predict_interval(self, X):
        self._require_fitted()
        a, b = self._lo.predict(X), self._hi.predict(X)
        return np.minimum(a, b), np.maximum(a, b)

    def evaluate(self, X, y):
        self._require_fitted()
        pred = self.predict(X)
        lo, hi = self.predict_interval(X)
        venue_pred = X["venue_avg_first"].fillna(self._train_mean)
        return {
            "model": self.selected,
            "MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": r2_score(y, pred),
            "80% range coverage": float(np.mean((y >= lo) & (y <= hi))),
            "baseline MAE (always train mean)": mean_absolute_error(y, np.full(len(y), self._train_mean)),
            "baseline MAE (venue average)": mean_absolute_error(y, venue_pred),
        }


class WinnerPredictor(BasePredictor):
    """Classification: P(team batting first wins). Logistic regression vs gradient boosting."""

    def __init__(self):
        super().__init__(name="Winner Predictor (Classification)")
        self._majority = None

    @staticmethod
    def _candidates():
        return {
            "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                 LogisticRegression(C=0.5, max_iter=1000)),
            "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                l2_regularization=1.0, random_state=0),
        }

    def fit(self, X, y):
        y = y.astype(int)
        cut = self._split(X)
        self.validation = {}
        for name, est in self._candidates().items():
            est.fit(X.iloc[:cut], y.iloc[:cut])
            p = est.predict_proba(X.iloc[cut:])[:, 1]
            self.validation[name] = log_loss(y.iloc[cut:], p, labels=[0, 1])
        self.selected = min(self.validation, key=self.validation.get)
        self._model = self._candidates()[self.selected].fit(X, y)
        self._majority = int(y.mean() >= 0.5)
        self._is_fitted = True
        return self

    def predict(self, X):
        self._require_fitted()
        return self._model.predict(X)

    def predict_proba(self, X):
        """Probability that the team batting first wins."""
        self._require_fitted()
        return self._model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        self._require_fitted()
        y = y.astype(int)
        p = self.predict_proba(X)
        pred = (p >= 0.5).astype(int)
        return {
            "model": self.selected,
            "Accuracy": accuracy_score(y, pred),
            "F1": f1_score(y, pred),
            "Log loss": log_loss(y, p, labels=[0, 1]),
            "Brier score": brier_score_loss(y, p),
            "baseline accuracy (always majority class)": accuracy_score(y, np.full(len(y), self._majority)),
            "baseline accuracy (higher Elo wins)": accuracy_score(y, (X["elo_diff"] > 0).astype(int)),
        }

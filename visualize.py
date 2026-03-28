"""
FinanceAnomalyDetector — high-level pipeline.

Wraps FinanceFeatureEngineer + IsolationForest into a single
fit/predict interface and adds human-readable output.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

from .isolation_forest import IsolationForest
from .features import FinanceFeatureEngineer


@dataclass
class AnomalyResult:
    transaction_id: int | str
    date: str
    description: str
    amount: float
    category: str
    hour: int
    anomaly_score: float
    is_anomaly: bool
    risk_level: str  # "HIGH" | "MEDIUM" | "LOW"

    def __repr__(self) -> str:
        flag = "⚠ ANOMALY" if self.is_anomaly else "✓ Normal"
        return (
            f"[{flag}] {self.date} | {self.description[:30]:<30} | "
            f"₸{abs(self.amount):>10,.0f} | score={self.anomaly_score:.4f} | {self.risk_level}"
        )


class FinanceAnomalyDetector:
    """
    End-to-end anomaly detector for personal finance data.

    Usage
    -----
    >>> detector = FinanceAnomalyDetector(contamination=0.10)
    >>> results = detector.fit_predict(df)
    >>> anomalies = [r for r in results if r.is_anomaly]

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalous transactions (default 0.10 = 10%).
    n_estimators : int
        Number of trees in the isolation forest.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.10,
        n_estimators: int = 100,
        random_state: int | None = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._engineer = FinanceFeatureEngineer()
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FinanceAnomalyDetector":
        """Fit the feature engineer and isolation forest on historical data."""
        self._validate(df)
        X = self._engineer.fit_transform(df)
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Score new transactions. Returns list of AnomalyResult objects."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        self._validate(df)

        X = self._engineer.transform(df)
        scores = self._model.score_samples(X)
        predictions = self._model.predict(X)

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            score = float(scores[i])
            is_anomaly = bool(predictions[i])
            results.append(AnomalyResult(
                transaction_id=row.get("id", i),
                date=str(row.get("date", "")),
                description=str(row.get("description", "")),
                amount=float(row["amount"]),
                category=str(row.get("category", "Other")),
                hour=int(row.get("hour", 12)),
                anomaly_score=score,
                is_anomaly=is_anomaly,
                risk_level=self._risk_level(score),
            ))
        return results

    def fit_predict(self, df: pd.DataFrame) -> list[AnomalyResult]:
        """Fit and predict on the same dataset."""
        return self.fit(df).predict(df)

    def summary(self, results: list[AnomalyResult]) -> dict:
        """Return summary statistics for a list of results."""
        total = len(results)
        anomalies = [r for r in results if r.is_anomaly]
        return {
            "total_transactions": total,
            "anomalies_found": len(anomalies),
            "anomaly_rate": round(len(anomalies) / total, 4) if total else 0,
            "threshold": round(self._model.threshold_, 4),
            "top_anomalies": sorted(anomalies, key=lambda r: r.anomaly_score, reverse=True)[:5],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame):
        required = {"amount", "hour", "day_of_week", "category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.70:
            return "HIGH"
        if score >= 0.55:
            return "MEDIUM"
        return "LOW"

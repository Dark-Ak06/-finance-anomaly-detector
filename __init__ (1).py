"""
Feature engineering for financial transaction data.

Transforms raw transaction records into a numeric feature matrix
suitable for the Isolation Forest model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


CATEGORY_LIST = [
    "Food", "Shopping", "Transport", "Health",
    "Entertainment", "Salary", "Other",
]

FEATURE_NAMES = [
    "amount_normalized",
    "amount_zscore",
    "hour_normalized",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "category_encoded",
]


class FinanceFeatureEngineer:
    """
    Transforms raw transaction DataFrames into ML-ready feature matrices.

    Expected DataFrame columns:
        - amount      : float  (negative = expense, positive = income)
        - hour        : int    (0–23)
        - day_of_week : int    (0=Mon … 6=Sun)
        - category    : str    (one of CATEGORY_LIST)

    Cyclical encoding for `hour`:
        sin/cos encoding preserves the circular nature of time
        (23:00 is close to 00:00), which naive normalization loses.
    """

    def __init__(self):
        self._amount_mean: float = 0.0
        self._amount_std: float = 1.0
        self._amount_max: float = 1.0
        self._label_enc = LabelEncoder().fit(CATEGORY_LIST)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FinanceFeatureEngineer":
        amounts = df["amount"].abs()
        self._amount_mean = amounts.mean()
        self._amount_std = amounts.std() or 1.0
        self._amount_max = amounts.max() or 1.0
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        amounts = df["amount"].abs()
        hours = df["hour"].astype(float)

        features = np.column_stack([
            # 1. Amount normalized to [0, 1]
            amounts / self._amount_max,
            # 2. Z-score — catches unusually large transactions
            (amounts - self._amount_mean) / self._amount_std,
            # 3. Hour normalized to [0, 1]
            hours / 23.0,
            # 4–5. Cyclical hour encoding
            np.sin(2 * np.pi * hours / 24),
            np.cos(2 * np.pi * hours / 24),
            # 6. Day of week (0–6)
            df["day_of_week"].astype(float) / 6.0,
            # 7. Category label
            self._encode_categories(df["category"]) / (len(CATEGORY_LIST) - 1),
        ])
        return features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_categories(self, series: pd.Series) -> np.ndarray:
        # Map unknown categories to "Other"
        cleaned = series.where(series.isin(CATEGORY_LIST), other="Other")
        return self._label_enc.transform(cleaned).astype(float)

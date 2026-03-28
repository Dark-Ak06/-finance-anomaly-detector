"""
Unit tests for the Finance Anomaly Detector.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.isolation_forest import IsolationForest, IsolationTree, _c
from src.features import FinanceFeatureEngineer
from src.detector import FinanceAnomalyDetector
from src.data_loader import generate_sample_data, load_dataframe


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def make_df(n=50, seed=0):
    return generate_sample_data(n_normal=n, n_anomalies=5, seed=seed)


# ------------------------------------------------------------------ #
#  _c (average path length)                                           #
# ------------------------------------------------------------------ #

class TestC:
    def test_zero_for_one(self):
        assert _c(1) == 0.0

    def test_zero_for_zero(self):
        assert _c(0) == 0.0

    def test_positive_for_large(self):
        assert _c(100) > 0

    def test_monotone(self):
        values = [_c(n) for n in [2, 10, 50, 256]]
        assert values == sorted(values)


# ------------------------------------------------------------------ #
#  IsolationTree                                                       #
# ------------------------------------------------------------------ #

class TestIsolationTree:
    def test_fit_returns_self(self):
        X = np.random.rand(20, 3)
        tree = IsolationTree(max_depth=5)
        result = tree.fit(X)
        assert result is tree

    def test_path_length_positive(self):
        X = np.random.rand(30, 4)
        tree = IsolationTree(max_depth=6).fit(X)
        pl = tree.path_length(X[0])
        assert pl > 0

    def test_leaf_with_one_sample(self):
        X = np.array([[1.0, 2.0]])
        tree = IsolationTree(max_depth=5).fit(X)
        assert tree.is_leaf

    def test_identical_values_become_leaf(self):
        X = np.ones((10, 3))
        tree = IsolationTree(max_depth=5).fit(X)
        assert tree.is_leaf


# ------------------------------------------------------------------ #
#  IsolationForest                                                     #
# ------------------------------------------------------------------ #

class TestIsolationForest:
    def test_fit_sets_threshold(self):
        X = np.random.rand(100, 4)
        model = IsolationForest(n_estimators=20, contamination=0.1, random_state=0)
        model.fit(X)
        assert 0 < model.threshold_ < 1

    def test_score_samples_shape(self):
        X = np.random.rand(50, 4)
        model = IsolationForest(n_estimators=10, random_state=0).fit(X)
        scores = model.score_samples(X)
        assert scores.shape == (50,)

    def test_scores_in_range(self):
        X = np.random.rand(50, 4)
        model = IsolationForest(n_estimators=10, random_state=0).fit(X)
        scores = model.score_samples(X)
        assert scores.min() > 0
        assert scores.max() <= 1.0

    def test_predict_binary(self):
        X = np.random.rand(50, 4)
        model = IsolationForest(n_estimators=10, random_state=0).fit(X)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_contamination_rate(self):
        """Predictions should match contamination fraction approximately."""
        X = np.random.rand(200, 4)
        model = IsolationForest(n_estimators=50, contamination=0.10, random_state=42).fit(X)
        preds = model.predict(X)
        rate = preds.mean()
        assert 0.05 <= rate <= 0.20  # within ±10% of target

    def test_obvious_anomaly_detected(self):
        """A single extreme outlier should be flagged."""
        np.random.seed(0)
        normal = np.random.randn(100, 2) * 0.1
        outlier = np.array([[100.0, 100.0]])
        X = np.vstack([normal, outlier])
        model = IsolationForest(n_estimators=50, contamination=0.05, random_state=0).fit(X)
        assert model.predict(outlier)[0] == 1


# ------------------------------------------------------------------ #
#  FinanceFeatureEngineer                                              #
# ------------------------------------------------------------------ #

class TestFeatureEngineer:
    def test_output_shape(self):
        df = make_df(50)
        eng = FinanceFeatureEngineer()
        X = eng.fit_transform(df)
        assert X.shape == (len(df), 7)

    def test_no_nans(self):
        df = make_df(50)
        eng = FinanceFeatureEngineer()
        X = eng.fit_transform(df)
        assert not np.isnan(X).any()

    def test_transform_without_fit_raises(self):
        df = make_df(20)
        eng = FinanceFeatureEngineer()
        with pytest.raises(RuntimeError):
            eng.transform(df)

    def test_unknown_category_handled(self):
        df = make_df(20)
        df.loc[0, "category"] = "UnknownCategory"
        eng = FinanceFeatureEngineer()
        X = eng.fit_transform(df)
        assert not np.isnan(X).any()


# ------------------------------------------------------------------ #
#  FinanceAnomalyDetector                                              #
# ------------------------------------------------------------------ #

class TestDetector:
    def test_fit_predict_returns_results(self):
        df = make_df(50)
        detector = FinanceAnomalyDetector(contamination=0.10, random_state=0)
        results = detector.fit_predict(df)
        assert len(results) == len(df)

    def test_all_have_scores(self):
        df = make_df(50)
        detector = FinanceAnomalyDetector(random_state=0)
        results = detector.fit_predict(df)
        assert all(0 <= r.anomaly_score <= 1 for r in results)

    def test_risk_levels_valid(self):
        df = make_df(50)
        detector = FinanceAnomalyDetector(random_state=0)
        results = detector.fit_predict(df)
        assert all(r.risk_level in ("HIGH", "MEDIUM", "LOW") for r in results)

    def test_predict_without_fit_raises(self):
        df = make_df(20)
        detector = FinanceAnomalyDetector()
        with pytest.raises(RuntimeError):
            detector.predict(df)

    def test_missing_column_raises(self):
        df = make_df(20).drop(columns=["amount"])
        detector = FinanceAnomalyDetector()
        with pytest.raises(ValueError):
            detector.fit(df)

    def test_summary_keys(self):
        df = make_df(50)
        detector = FinanceAnomalyDetector(random_state=0)
        results = detector.fit_predict(df)
        summary = detector.summary(results)
        assert "total_transactions" in summary
        assert "anomalies_found" in summary
        assert "anomaly_rate" in summary
        assert "top_anomalies" in summary

    def test_reproducible_with_seed(self):
        df = make_df(60)
        r1 = FinanceAnomalyDetector(random_state=7).fit_predict(df)
        r2 = FinanceAnomalyDetector(random_state=7).fit_predict(df)
        scores1 = [r.anomaly_score for r in r1]
        scores2 = [r.anomaly_score for r in r2]
        assert scores1 == scores2

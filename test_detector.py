"""
Isolation Forest — implementation from scratch.

Based on: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).
"Isolation forest." ICDM 2008.
"""

import numpy as np
from typing import Optional


class IsolationTree:
    """A single isolation tree."""

    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.split_feature: Optional[int] = None
        self.split_value: Optional[float] = None
        self.left: Optional["IsolationTree"] = None
        self.right: Optional["IsolationTree"] = None
        self.size: int = 0
        self.is_leaf: bool = False

    def fit(self, X: np.ndarray, depth: int = 0) -> "IsolationTree":
        self.size = len(X)

        if depth >= self.max_depth or len(X) <= 1:
            self.is_leaf = True
            return self

        n_features = X.shape[1]
        self.split_feature = np.random.randint(0, n_features)
        col = X[:, self.split_feature]

        min_val, max_val = col.min(), col.max()
        if min_val == max_val:
            self.is_leaf = True
            return self

        self.split_value = np.random.uniform(min_val, max_val)

        left_mask = col < self.split_value
        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1)
        return self

    def path_length(self, x: np.ndarray, depth: int = 0) -> float:
        if self.is_leaf:
            return depth + _c(self.size)
        if x[self.split_feature] < self.split_value:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


def _c(n: int) -> float:
    """Average path length of unsuccessful BST search."""
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class IsolationForest:
    """
    Isolation Forest for anomaly detection.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    max_samples : int or 'auto'
        Samples to draw for each tree. 'auto' = min(256, n_samples).
    contamination : float
        Expected proportion of anomalies in the dataset (0.0–0.5).
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        contamination: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees_: list[IsolationTree] = []
        self.threshold_: float = 0.5
        self._n_samples: int = 0

    def fit(self, X: np.ndarray) -> "IsolationForest":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = len(X)
        self._n_samples = (
            min(256, n) if self.max_samples == "auto" else int(self.max_samples)
        )
        max_depth = int(np.ceil(np.log2(self._n_samples)))

        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, size=self._n_samples, replace=False if n >= self._n_samples else True)
            tree = IsolationTree(max_depth).fit(X[idx])
            self.trees_.append(tree)

        scores = self.score_samples(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score for each sample. Higher = more anomalous."""
        avg_paths = np.array([
            np.mean([tree.path_length(x) for tree in self.trees_])
            for x in X
        ])
        return np.power(2.0, -avg_paths / _c(self._n_samples))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal."""
        return (self.score_samples(X) >= self.threshold_).astype(int)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

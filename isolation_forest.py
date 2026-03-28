{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔍 Finance Anomaly Detector — ML Exploration\n",
    "\n",
    "This notebook walks through the full pipeline:\n",
    "1. Data generation & EDA\n",
    "2. Feature engineering\n",
    "3. Isolation Forest training\n",
    "4. Results analysis & visualization\n",
    "\n",
    "**Algorithm:** Isolation Forest (implemented from scratch)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib\n",
    "matplotlib.rcParams['figure.figsize'] = (12, 5)\n",
    "\n",
    "from src import FinanceAnomalyDetector, generate_sample_data\n",
    "from src.features import FinanceFeatureEngineer\n",
    "from src.isolation_forest import IsolationForest\n",
    "from src.visualize import (\n",
    "    plot_anomaly_scatter,\n",
    "    plot_score_distribution,\n",
    "    plot_spending_timeline,\n",
    "    plot_category_breakdown,\n",
    ")\n",
    "\n",
    "print('✅ Imports OK')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Generate Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = generate_sample_data(n_normal=300, n_anomalies=30, seed=42)\n",
    "print(f'Dataset: {len(df)} transactions')\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic EDA\n",
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "\n",
    "df['amount'].abs().hist(bins=40, ax=axes[0], color='#6ee7b7', edgecolor='none')\n",
    "axes[0].set_title('Amount distribution')\n",
    "axes[0].set_xlabel('₸')\n",
    "\n",
    "df['hour'].hist(bins=24, ax=axes[1], color='#818cf8', edgecolor='none')\n",
    "axes[1].set_title('Transaction hour')\n",
    "axes[1].set_xlabel('Hour of day')\n",
    "\n",
    "df['category'].value_counts().plot(kind='bar', ax=axes[2], color='#fbbf24', edgecolor='none')\n",
    "axes[2].set_title('Category counts')\n",
    "axes[2].tick_params(axis='x', rotation=45)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Engineering\n",
    "\n",
    "Key design decision: **cyclical encoding** for hour of day.\n",
    "Without it, 23:00 and 00:00 look maximally different (23 vs 0). With sin/cos they wrap correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "engineer = FinanceFeatureEngineer()\n",
    "X = engineer.fit_transform(df)\n",
    "\n",
    "print(f'Feature matrix shape: {X.shape}')\n",
    "print(f'Features: {engineer.feature_names}')\n",
    "print(f'\\nStats:')\n",
    "pd.DataFrame(X, columns=engineer.feature_names).describe().round(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize cyclical hour encoding\n",
    "hours = np.arange(0, 24)\n",
    "sin_enc = np.sin(2 * np.pi * hours / 24)\n",
    "cos_enc = np.cos(2 * np.pi * hours / 24)\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "ax1.plot(hours, sin_enc, 'o-', color='#6ee7b7', label='sin')\n",
    "ax1.plot(hours, cos_enc, 's-', color='#818cf8', label='cos')\n",
    "ax1.set_xlabel('Hour'); ax1.set_title('Cyclical encoding of hour')\n",
    "ax1.legend(); ax1.grid(alpha=0.3)\n",
    "\n",
    "ax2.scatter(sin_enc, cos_enc, c=hours, cmap='plasma', s=60)\n",
    "plt.colorbar(ax2.collections[0], ax=ax2, label='Hour')\n",
    "ax2.set_xlabel('sin(hour)'); ax2.set_ylabel('cos(hour)')\n",
    "ax2.set_title('Hour on unit circle — 23:00 ≈ 00:00')\n",
    "ax2.grid(alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Train Isolation Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "detector = FinanceAnomalyDetector(\n",
    "    contamination=0.10,\n",
    "    n_estimators=100,\n",
    "    random_state=42,\n",
    ")\n",
    "results = detector.fit_predict(df)\n",
    "summary = detector.summary(results)\n",
    "\n",
    "print(f\"Total transactions : {summary['total_transactions']}\")\n",
    "print(f\"Anomalies found    : {summary['anomalies_found']}\")\n",
    "print(f\"Anomaly rate       : {summary['anomaly_rate']*100:.1f}%\")\n",
    "print(f\"Decision threshold : {summary['threshold']}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('\\n⚠  Top anomalies:')\n",
    "for r in summary['top_anomalies']:\n",
    "    print(f'  {r}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Visualize Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plot_anomaly_scatter(results)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plot_score_distribution(results, threshold=summary['threshold'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plot_spending_timeline(results)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig = plot_category_breakdown(results)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Compare with sklearn\n",
    "\n",
    "Let's verify our implementation produces similar results to sklearn's IsolationForest."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import IsolationForest as SklearnIF\n",
    "from src.isolation_forest import IsolationForest as OurIF\n",
    "\n",
    "X_test = np.random.randn(200, 4)\n",
    "X_test = np.vstack([X_test, np.array([[10, 10, 10, 10]] * 10)])  # inject outliers\n",
    "\n",
    "# Our implementation\n",
    "our_model = OurIF(n_estimators=100, contamination=0.05, random_state=42).fit(X_test)\n",
    "our_scores = our_model.score_samples(X_test)\n",
    "\n",
    "# sklearn\n",
    "sk_model = SklearnIF(n_estimators=100, contamination=0.05, random_state=42).fit(X_test)\n",
    "# sklearn scores are negated, rescale to [0,1] for comparison\n",
    "sk_raw = -sk_model.score_samples(X_test)\n",
    "sk_scores = (sk_raw - sk_raw.min()) / (sk_raw.max() - sk_raw.min())\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
    "ax1.scatter(range(len(our_scores)), our_scores, s=10, alpha=0.6, c='#6ee7b7')\n",
    "ax1.set_title('Our Isolation Forest scores')\n",
    "ax1.set_ylabel('Anomaly score')\n",
    "ax2.scatter(range(len(sk_scores)), sk_scores, s=10, alpha=0.6, c='#818cf8')\n",
    "ax2.set_title('sklearn Isolation Forest scores (normalized)')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "corr = np.corrcoef(our_scores, sk_scores)[0, 1]\n",
    "print(f'Pearson correlation between implementations: {corr:.4f}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

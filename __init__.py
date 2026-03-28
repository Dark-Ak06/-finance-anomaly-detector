"""
Data loading utilities for the finance anomaly detector.
Supports CSV files and synthetic data generation for demos.
"""

import pandas as pd
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = {"date", "description", "amount", "category"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load transactions from a CSV file and add derived columns.

    Expected CSV columns:
        date, description, amount, category

    Optional columns (auto-derived if missing):
        hour, day_of_week
    """
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = _add_time_features(df)
    df["id"] = df.get("id", pd.Series(range(len(df))))
    return df


def load_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare an existing DataFrame (adds time features if absent)."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    return _add_time_features(df.copy())


def generate_sample_data(
    n_normal: int = 200,
    n_anomalies: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transaction data for demos and testing.

    Normal transactions follow realistic patterns (e.g. groceries
    during daytime, salaries on workdays). Anomalies are injected
    as large-amount, odd-hour transactions.
    """
    rng = np.random.default_rng(seed)

    category_profiles = {
        "Food":          {"amount": (500, 4000),   "hours": (8, 21)},
        "Shopping":      {"amount": (1000, 25000),  "hours": (10, 20)},
        "Transport":     {"amount": (200, 2500),    "hours": (7, 22)},
        "Health":        {"amount": (1000, 12000),  "hours": (9, 18)},
        "Entertainment": {"amount": (500, 7000),    "hours": (12, 23)},
        "Salary":        {"amount": (150000, 350000), "hours": (9, 17)},
        "Other":         {"amount": (300, 8000),    "hours": (8, 20)},
    }

    records = []
    categories = list(category_profiles.keys())

    # Normal transactions
    for i in range(n_normal):
        cat = rng.choice(categories)
        profile = category_profiles[cat]
        amount_range = profile["amount"]
        hour_range = profile["hours"]

        days_ago = rng.integers(0, 30)
        date = pd.Timestamp.today() - pd.Timedelta(days=int(days_ago))
        hour = int(rng.integers(*hour_range))

        records.append({
            "id": i,
            "date": date.strftime("%Y-%m-%d"),
            "description": _random_description(cat, rng),
            "amount": -round(rng.uniform(*amount_range), 2),  # negative = expense
            "category": cat,
            "hour": hour,
            "day_of_week": date.dayofweek,
        })

    # Injected anomalies
    anomaly_descs = [
        "SUSPICIOUS WIRE TRANSFER", "Unknown merchant 3AM",
        "Crypto exchange large", "ATM withdrawal abroad",
        "International payment", "Duplicate charge",
        "Unusual casino transaction", "Unrecognised subscription",
        "Dark web market", "Large gambling site",
    ]
    for j in range(n_anomalies):
        days_ago = rng.integers(0, 30)
        date = pd.Timestamp.today() - pd.Timedelta(days=int(days_ago))
        records.append({
            "id": n_normal + j,
            "date": date.strftime("%Y-%m-%d"),
            "description": anomaly_descs[j % len(anomaly_descs)],
            "amount": -round(rng.uniform(80_000, 600_000), 2),
            "category": rng.choice(["Shopping", "Other", "Entertainment"]),
            "hour": int(rng.integers(0, 5)),   # suspicious: deep night
            "day_of_week": date.dayofweek,
        })

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

_DESCRIPTIONS = {
    "Food":          ["Magnum Market", "KazMart", "Döner Kebab", "Coffee Lab", "Chapan supermarket", "Ramstore"],
    "Shopping":      ["Sulpak electronics", "Megaphone", "Wildberries KZ", "IPORT", "H&M Almaty"],
    "Transport":     ["Yandex Go", "Bolt taxi", "Astana Airlines", "Bus ticket", "Parking meter"],
    "Health":        ["City Clinic", "Pharmacy 36.6", "Medline lab", "Dentist", "Fitlife gym"],
    "Entertainment": ["Chaplin Cinema", "Steam", "Netflix KZ", "Concert ticket", "Billiard club"],
    "Salary":        ["Salary deposit", "Bonus Q1", "Freelance payment", "Consulting fee"],
    "Other":         ["Utility bill", "Internet Beeline", "Mobile top-up", "Insurance", "Bank transfer"],
}


def _random_description(category: str, rng: np.random.Generator) -> str:
    options = _DESCRIPTIONS.get(category, ["Transaction"])
    return str(rng.choice(options))


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "hour" not in df.columns:
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["hour"] = df["date"].dt.hour
        else:
            df["hour"] = 12  # fallback
    if "day_of_week" not in df.columns:
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["day_of_week"] = df["date"].dt.dayofweek
        else:
            df["day_of_week"] = 0
    return df

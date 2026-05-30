from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


RISK_LABELS = ["Low", "Moderate", "High"]


@dataclass(frozen=True)
class Features:
    avg_social_hours_per_day: float
    sessions_per_day: float
    before_bed_hours: float
    cannot_cut_down: int
    affects_sleep: int
    impacts_work_study: int
    craving_irritability: int


def _clip01(x) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def score_addiction(feat: Features) -> float:
    """
    Returns an addiction score in [0, 100].

    This is an interpretable heuristic score used both for the dashboard and
    as ground-truth for the synthetic ML training data.
    """
    # Expected input ranges (we clamp defensively).
    avg_h = float(np.clip(feat.avg_social_hours_per_day, 0, 12))
    sessions = float(np.clip(feat.sessions_per_day, 0, 30))
    before_bed = float(np.clip(feat.before_bed_hours, 0, 6))

    cannot_cut_down = int(np.clip(feat.cannot_cut_down, 0, 1))
    affects_sleep = int(np.clip(feat.affects_sleep, 0, 1))
    impacts_work_study = int(np.clip(feat.impacts_work_study, 0, 1))
    craving_irritability = int(np.clip(feat.craving_irritability, 0, 1))

    # Weighted sum (roughly: time + compulsive behavior).
    raw = (
        0.9 * avg_h
        + 0.55 * sessions
        + 1.0 * before_bed
        + 18.0 * cannot_cut_down
        + 14.0 * affects_sleep
        + 10.0 * impacts_work_study
        + 9.0 * craving_irritability
    )

    # Max possible score under the clamped ranges.
    max_raw = (
        0.9 * 12
        + 0.55 * 30
        + 1.0 * 6
        + 18.0 * 1
        + 14.0 * 1
        + 10.0 * 1
        + 9.0 * 1
    )
    return float(np.clip((raw / max_raw) * 100.0, 0.0, 100.0))


def score_to_class(score: float) -> int:
    # Thresholds tuned to make each class non-trivial on synthetic data.
    if score < 35.0:
        return 0
    if score < 65.0:
        return 1
    return 2


def features_to_vector(feat: Features) -> np.ndarray:
    return np.array(
        [
            feat.avg_social_hours_per_day,
            feat.sessions_per_day,
            feat.before_bed_hours,
            feat.cannot_cut_down,
            feat.affects_sleep,
            feat.impacts_work_study,
            feat.craving_irritability,
        ],
        dtype=float,
    )


def vector_to_features(x: np.ndarray) -> Features:
    return Features(
        avg_social_hours_per_day=float(x[0]),
        sessions_per_day=float(x[1]),
        before_bed_hours=float(x[2]),
        cannot_cut_down=int(round(x[3])),
        affects_sleep=int(round(x[4])),
        impacts_work_study=int(round(x[5])),
        craving_irritability=int(round(x[6])),
    )


def generate_synthetic(n: int = 4000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    avg_hours = rng.uniform(0, 10.5, size=n)
    sessions = rng.uniform(0, 16, size=n)
    before_bed = rng.uniform(0, 3.5, size=n)

    # Compulsive indicators correlated with time + before-bed usage.
    cannot_cut_down_p = _clip01(0.05 + 0.07 * (avg_hours / 10) + 0.25 * (before_bed / 3.5))
    affects_sleep_p = _clip01(0.03 + 0.22 * (before_bed / 3.5) + 0.18 * (avg_hours / 10))
    impacts_work_p = _clip01(0.02 + 0.14 * (avg_hours / 10) + 0.05 * (sessions / 16))
    craving_p = _clip01(0.02 + 0.10 * (sessions / 16) + 0.18 * (cannot_cut_down_p))

    cannot_cut_down = rng.binomial(1, cannot_cut_down_p).astype(int)
    affects_sleep = rng.binomial(1, affects_sleep_p).astype(int)
    impacts_work_study = rng.binomial(1, impacts_work_p).astype(int)
    craving_irritability = rng.binomial(1, craving_p).astype(int)

    X = np.column_stack([avg_hours, sessions, before_bed, cannot_cut_down, affects_sleep, impacts_work_study, craving_irritability])

    # Labels from our heuristic score -> 3-class bucket.
    y = np.empty(n, dtype=int)
    for i in range(n):
        feat = vector_to_features(X[i])
        s = score_addiction(feat)
        y[i] = score_to_class(s)

    return X, y


def train_risk_model(
    n_samples: int = 4500, seed: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    X, y = generate_synthetic(n=n_samples, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)

    model = RandomForestClassifier(
        n_estimators=220,
        random_state=seed,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=RISK_LABELS, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "classification_report": report,
        "class_balance_train": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
    }
    return model, metrics


def predict_risk_proba(model: RandomForestClassifier, feat: Features) -> np.ndarray:
    x = features_to_vector(feat).reshape(1, -1)
    proba = model.predict_proba(x)[0]  # shape: (3,)
    # Safety normalization (some edge cases shouldn't happen).
    proba = proba / np.sum(proba)
    return proba


def class_from_proba(proba: np.ndarray) -> int:
    return int(np.argmax(proba))


def compute_score_from_inputs(
    avg_social_hours_per_day: float,
    sessions_per_day: float,
    before_bed_hours: float,
    cannot_cut_down: bool,
    affects_sleep: bool,
    impacts_work_study: bool,
    craving_irritability: bool,
) -> float:
    feat = Features(
        avg_social_hours_per_day=avg_social_hours_per_day,
        sessions_per_day=sessions_per_day,
        before_bed_hours=before_bed_hours,
        cannot_cut_down=int(cannot_cut_down),
        affects_sleep=int(affects_sleep),
        impacts_work_study=int(impacts_work_study),
        craving_irritability=int(craving_irritability),
    )
    return score_addiction(feat)


def try_extract_time_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Attempts to extract a daily time series of total social screen time.

    Supports:
    - columns: date, platform, hours  (long format)
    - columns: date, hours (already daily-ish)
    - columns: date, social_hours
    """
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        return None

    date_col = cols["date"]
    df2 = df.copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
    df2 = df2.dropna(subset=[date_col])
    if df2.empty:
        return None

    # Determine hours column.
    if "hours" in cols and ("platform" in cols or df2.shape[0] > 0):
        hours_col = cols["hours"]
        # If platform exists, sum across platforms; else just group by date.
        if "platform" in cols:
            daily = df2.groupby(date_col)[hours_col].sum().sort_index()
        else:
            daily = df2.groupby(date_col)[hours_col].sum().sort_index()
        return daily

    if "social_hours" in cols:
        social_hours_col = cols["social_hours"]
        daily = df2.groupby(date_col)[social_hours_col].sum().sort_index()
        return daily

    return None


def extract_platform_breakdown(df: pd.DataFrame) -> Optional[pd.Series]:
    cols = {c.lower(): c for c in df.columns}
    if "platform" not in cols:
        return None

    if "hours" in cols:
        platform_col = cols["platform"]
        hours_col = cols["hours"]
        series = df.groupby(platform_col)[hours_col].sum().sort_values(ascending=False)
        return series

    if "social_hours" in cols:
        platform_col = cols["platform"]
        social_hours_col = cols["social_hours"]
        series = df.groupby(platform_col)[social_hours_col].sum().sort_values(ascending=False)
        return series

    return None


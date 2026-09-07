"""
analytics.py
------------
Feature engineering, statistical analysis, and a lightweight ML model
for predicting video performance. This is what turns raw API data into
actual data-science output instead of just a fetch-and-plot script.
"""

import re
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


# ----------------------------------------------------------------------------
# Duration parsing (YouTube returns ISO 8601, e.g. "PT4M13S")
# ----------------------------------------------------------------------------
def parse_iso8601_duration(duration: str) -> int:
    """Convert 'PT1H2M3S' style duration into total seconds."""
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration
    )
    if not match:
        return 0
    hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    return hours * 3600 + minutes * 60 + seconds


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns that are useful for both visualization and modeling:
    - duration_seconds, duration_minutes
    - is_short (< 60s, YouTube Shorts heuristic)
    - day_of_week, hour_published (posting time patterns)
    - title_length, title_word_count
    - title_has_number, title_has_question, title_is_allcaps_word (clickbait signals)
    - views_per_day (normalizes for how long a video's been live)
    """
    df = df.copy()

    df["duration_seconds"] = df["duration"].apply(parse_iso8601_duration)
    df["duration_minutes"] = (df["duration_seconds"] / 60).round(2)
    df["is_short"] = df["duration_seconds"] <= 60

    df["day_of_week"] = df["published_at"].dt.day_name()
    df["hour_published"] = df["published_at"].dt.hour

    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["title_has_number"] = df["title"].str.contains(r"\d", regex=True)
    df["title_has_question"] = df["title"].str.contains(r"\?", regex=True)
    df["title_has_allcaps_word"] = df["title"].apply(
        lambda t: any(w.isupper() and len(w) > 1 for w in t.split())
    )

    days_live = (pd.Timestamp.now(tz="UTC") - df["published_at"]).dt.days.clip(lower=1)
    df["views_per_day"] = (df["views"] / days_live).round(1)

    return df


# ----------------------------------------------------------------------------
# Outlier / viral video detection
# ----------------------------------------------------------------------------
def flag_outliers(df: pd.DataFrame, z_threshold: float = 1.5) -> pd.DataFrame:
    """Flags videos whose views are unusually high/low relative to the channel's mean (z-score)."""
    df = df.copy()
    mean, std = df["views"].mean(), df["views"].std()
    if std == 0 or pd.isna(std):
        df["view_zscore"] = 0.0
    else:
        df["view_zscore"] = ((df["views"] - mean) / std).round(2)
    df["is_viral_outlier"] = df["view_zscore"] >= z_threshold
    df["is_underperformer"] = df["view_zscore"] <= -z_threshold
    return df


# ----------------------------------------------------------------------------
# Best posting time analysis
# ----------------------------------------------------------------------------
def best_posting_times(df: pd.DataFrame) -> pd.DataFrame:
    """Average views grouped by day-of-week, for a heatmap-style view."""
    grouped = (
        df.groupby("day_of_week")["views"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_views", "count": "video_count"})
        .round(0)
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = grouped.reindex([d for d in day_order if d in grouped.index])
    return grouped.reset_index()


# ----------------------------------------------------------------------------
# ML: predict views from title/posting features
# ----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "title_length",
    "title_word_count",
    "title_has_number",
    "title_has_question",
    "title_has_allcaps_word",
    "duration_seconds",
    "hour_published",
]


def train_view_predictor(df: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict, pd.DataFrame]:
    """
    Trains a RandomForestRegressor to predict view count from title/format
    features. Returns (model, metrics_dict, feature_importance_df).

    Uses log1p(views) as the target since view counts are heavily
    right-skewed (a few viral videos dominate raw scale).

    Evaluation uses k-fold cross-validation rather than a single
    train/test split: with only 10-100 videos, a single held-out split
    can put as few as 3-8 videos in the test set, which makes R² swing
    wildly between runs on the *same* channel and gives a misleading
    read on how good the model actually is. Cross-validation averages
    over multiple splits, so the reported R² is far more stable and
    trustworthy. k is scaled down automatically for very small samples.
    Requires at least 8 videos to produce a meaningful evaluation.
    """
    if len(df) < 8:
        raise ValueError(
            f"Need at least 8 videos to train a model reliably, got {len(df)}. "
            "Try pulling more videos."
        )

    X = df[FEATURE_COLUMNS].copy()
    for col in ["title_has_number", "title_has_question", "title_has_allcaps_word"]:
        X[col] = X[col].astype(int)

    y = np.log1p(df["views"])

    # Scale folds down for small samples (need >=2 samples per fold).
    n_folds = min(5, max(2, len(df) // 4))
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    r2_scores = cross_val_score(cv_model, X, y, cv=kfold, scoring="r2")
    mae_scores = -cross_val_score(
        cv_model, X, y, cv=kfold, scoring="neg_mean_absolute_error"
    )

    metrics = {
        "r2": round(float(np.mean(r2_scores)), 3),
        "r2_std": round(float(np.std(r2_scores)), 3),
        "mae_log_scale": round(float(np.mean(mae_scores)), 3),
        "n_folds": n_folds,
        "n_samples": len(df),
    }

    # Final model trained on the *full* dataset (cross-validation above was
    # only for evaluation) so predict_views() gets the benefit of every
    # video, not just one fold's worth.
    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, metrics, importance_df


def predict_views(model: RandomForestRegressor, title: str, duration_seconds: int, hour_published: int) -> int:
    """Predict expected views for a hypothetical new video."""
    features = pd.DataFrame([{
        "title_length": len(title),
        "title_word_count": len(title.split()),
        "title_has_number": int(bool(re.search(r"\d", title))),
        "title_has_question": int("?" in title),
        "title_has_allcaps_word": int(any(w.isupper() and len(w) > 1 for w in title.split())),
        "duration_seconds": duration_seconds,
        "hour_published": hour_published,
    }])
    log_pred = model.predict(features)[0]
    return int(np.expm1(log_pred))

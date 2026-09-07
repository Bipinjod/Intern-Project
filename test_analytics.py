"""
test_analytics.py
------------------
Unit tests for the pure functions in analytics.py. These don't touch the
YouTube API — they run on small synthetic DataFrames so they're fast and
don't burn API quota.

Run with: pytest test_analytics.py -v
"""

import pandas as pd
import pytest

from analytics import parse_iso8601_duration, add_engineered_features, flag_outliers


# ----------------------------------------------------------------------------
# parse_iso8601_duration
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "duration_str,expected_seconds",
    [
        ("PT4M13S", 4 * 60 + 13),
        ("PT1H2M3S", 1 * 3600 + 2 * 60 + 3),
        ("PT45S", 45),
        ("PT1H", 3600),
        ("", 0),
        ("not-a-duration", 0),
    ],
)
def test_parse_iso8601_duration(duration_str, expected_seconds):
    assert parse_iso8601_duration(duration_str) == expected_seconds


# ----------------------------------------------------------------------------
# add_engineered_features
# ----------------------------------------------------------------------------
def _sample_df():
    return pd.DataFrame({
        "title": [
            "How I Made 10000 Dollars in a Day",   # has number
            "Is this the best phone ever?",         # has question
            "I Tried SHOUTING for 24 Hours",        # has allcaps word
            "quiet vlog",                           # plain
        ],
        "published_at": pd.to_datetime([
            "2026-01-01T10:00:00Z",
            "2026-01-02T18:00:00Z",
            "2026-01-03T05:00:00Z",
            "2026-01-04T22:00:00Z",
        ]),
        "duration": ["PT4M13S", "PT30S", "PT10M0S", "PT45S"],
        "views": [1000, 2000, 3000, 4000],
        "likes": [10, 20, 30, 40],
        "comments": [1, 2, 3, 4],
    })


def test_add_engineered_features_basic_columns():
    df = add_engineered_features(_sample_df())

    # duration parsing
    assert df.loc[0, "duration_seconds"] == 253
    assert df.loc[1, "duration_seconds"] == 30

    # is_short: <= 60 seconds
    assert df.loc[1, "is_short"] == True   # 30s
    assert df.loc[3, "is_short"] == True   # 45s
    assert df.loc[0, "is_short"] == False  # 253s
    assert df.loc[2, "is_short"] == False  # 600s


def test_add_engineered_features_title_signals():
    df = add_engineered_features(_sample_df())

    assert df.loc[0, "title_has_number"] == True
    assert df.loc[3, "title_has_number"] == False

    assert df.loc[1, "title_has_question"] == True
    assert df.loc[0, "title_has_question"] == False

    assert df.loc[2, "title_has_allcaps_word"] == True   # "SHOUTING"
    assert df.loc[3, "title_has_allcaps_word"] == False  # "quiet vlog"

    # word count sanity check
    assert df.loc[3, "title_word_count"] == 2  # "quiet vlog"


# ----------------------------------------------------------------------------
# flag_outliers
# ----------------------------------------------------------------------------
def test_flag_outliers_detects_viral_video():
    df = pd.DataFrame({
        "views": [100, 110, 90, 105, 95, 10000],  # last one is a clear outlier
    })
    result = flag_outliers(df, z_threshold=1.5)

    assert result.loc[5, "is_viral_outlier"] == True
    assert result.loc[0, "is_viral_outlier"] == False


def test_flag_outliers_handles_zero_variance():
    # All identical views -> std is 0, must not raise or produce NaN z-scores
    df = pd.DataFrame({"views": [500, 500, 500]})
    result = flag_outliers(df)

    assert (result["view_zscore"] == 0.0).all()
    assert not result["is_viral_outlier"].any()

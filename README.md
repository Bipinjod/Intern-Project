# YouTube Analytics Dashboard

Data science internship project completed at Prime Times Television as
part of the Junior Data Scientist internship. The project delivers an
interactive Streamlit dashboard that pulls, analyzes, and visualizes
performance data for any public YouTube channel using the YouTube Data
API v3, and includes a machine-learning module for estimating expected
video performance from title and format features.

## Weekly Breakdown

| Week | Focus |
|------|-------|
| Week 1–2 | Internship onboarding; learned the YouTube Data API v3 (quotas, endpoints, auth); scoped the dashboard requirements with company supervisor |
| Week 3–4 | Built `youtube_api.py` — channel resolution (URL/handle/name), channel metadata fetching, paginated video/stats retrieval with quota-conscious batching |
| Week 5–6 | Built `analytics.py` — feature engineering (duration parsing, title signals, posting-time features), engagement rate calculation, z-score outlier detection |
| Week 7–8 | Built the machine learning module — `RandomForestRegressor` view predictor on `log1p`-transformed views, cross-validated evaluation, feature importance analysis |
| Week 9–10 | Built `app.py` — full Streamlit dashboard UI, custom dark theme, tabbed layout (Overview / Deep Analytics / Performance Predictor), interactive charts |
| Week 11 | Testing and debugging — wrote unit tests (`test_analytics.py`), fixed edge-case bugs (zero-view videos, hidden likes/disabled comments), added error handling for API failures |
| Week 12 | Final polish — GitHub repository setup, documentation, live testing against real channels, internship report preparation |

## About the Project

The dashboard analyzes a YouTube channel's most recent uploads and turns
raw API data into structured insight: channel growth metrics, per-video
engagement, posting-time patterns, format analysis, and outlier detection
for viral or underperforming videos. A Random Forest model, evaluated with
k-fold cross-validation, estimates expected views for a hypothetical new
upload based on that channel's own historical patterns.

## Objective

Build a reliable, quota-conscious analytics tool that a media company can
point at any public YouTube channel — its own or a competitor's — and get
back structured, statistically sound insight into what drives that
channel's performance, without needing to write any code or touch the raw
API directly.

## Workflow

1. **Channel Resolution** – Accepts a channel URL, `@handle`, or plain
   name and resolves it to a canonical channel ID via the YouTube Data
   API v3, with a search-based fallback for ambiguous input.
2. **Data Collection** – Pulls channel-level stats (subscribers, total
   views, upload count) and per-video stats (views, likes, comments,
   duration, publish time) for the N most recent uploads, batching
   requests to stay within API limits and caching responses to conserve
   daily quota.
3. **Data Cleaning & Feature Engineering** – Parses ISO 8601 durations,
   derives title/format signals (length, word count, question marks,
   numerals, all-caps words), computes day-of-week and hour-published
   features, and flags videos with hidden like counts or disabled
   comments so they don't distort engagement metrics.
4. **Statistical Analysis** – Computes engagement rate, views-per-day,
   z-score-based outlier detection for viral/underperforming videos, and
   correlation analysis between format features and view count.
5. **Machine Learning** – Trains a `RandomForestRegressor` on `log1p`-
   transformed view counts, evaluated with k-fold cross-validation
   (scaled to sample size) rather than a single train/test split, since
   small per-channel samples make a single split unreliable.
6. **Visualization & Reporting** – Presents all of the above through an
   interactive, tabbed Streamlit dashboard (Overview, Deep Analytics,
   Performance Predictor), with CSV export of the underlying data.

## Features

- Resolves a channel from a URL, handle, or name
- Pulls channel-level stats (subscribers, total views, video count)
- Pulls the N most recent uploads and their view/like/comment counts
- Computes engagement rate: `(likes + comments) / views * 100`, correctly
  excluding videos with hidden likes or disabled comments
- Views trend, top-videos, and engagement-distribution charts
- Outlier detection for viral and underperforming videos (z-score based)
- Best-day-to-post and duration/title-length format analysis
- Correlation analysis between format features and views
- ML-based view predictor with cross-validated accuracy reporting and
  feature importance breakdown
- CSV export of all pulled video data
- Friendly error handling for API quota limits, invalid keys, and
  unresolvable channels

## Repository Structure

```
Intern-Project/
├── app.py                # Streamlit UI, charts, and dashboard layout
├── youtube_api.py         # YouTube Data API v3 wrapper (channel
│                          # resolution, video/stat fetching)
├── analytics.py            # Feature engineering, outlier detection,
│                          # and the cross-validated ML predictor
├── test_analytics.py       # Unit tests for analytics.py
├── requirements.txt
├── .env.example
└── README.md
```

## Tools Used

Python (pandas, numpy, scikit-learn), Streamlit, Plotly, YouTube Data
API v3, pytest, Git/GitHub

## Skills Gained

REST API integration and quota management, data cleaning and feature
engineering, statistical analysis (z-score outlier detection,
correlation analysis), machine learning model evaluation with
cross-validation, interactive dashboard development, unit testing, and
version control workflow.

## Known Limitations

- The view predictor only uses title/format features (length, word
  count, question/number/allcaps signals, duration, upload hour) — no
  thumbnail or topic/category data yet, so predictive power is limited.
  Model accuracy is reported via k-fold cross-validation (not a single
  train/test split) since sample sizes of 10–100 videos make a single
  split unstable; expect a wide `r2_std` on small samples, and treat
  predictions as directional signals rather than precise estimates.
- Engagement rate excludes videos where a creator has hidden the like
  count or disabled comments, rather than treating those as genuine
  zero-engagement videos. If every video on a channel has hidden likes,
  the "Avg. engagement" metric shows "N/A" instead of a number.
- Only a snapshot of the channel's most recent uploads is analyzed —
  there's no historical tracking of how a channel's stats change over
  time.

## Planned Next Steps

- Add a "compare two channels" mode
- Add thumbnail and topic/category features to the ML model
- Persist historical pulls so you can track growth over time, not just a
  snapshot of current stats

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your API key**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and paste your YouTube Data API v3 key:
   ```
   YOUTUBE_API_KEY=your_actual_key_here
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. In the sidebar, enter a channel URL, `@handle`, or channel name and
   click **Analyze Channel**.

## Notes on API Quota

Free tier = 10,000 units/day. Each `videos.list` / `channels.list` call
is ~1 unit; `search.list` calls (used only as a fallback for channel
resolution) cost 100 units. Pulling 50 videos for a channel typically
costs well under 50 units, so there's plenty of room to experiment.

## Author

Bipin Dhakal
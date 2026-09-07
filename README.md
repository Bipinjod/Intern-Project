# YouTube Analytics Dashboard

Internship project — Junior Data Scientist, Prime Times Television.

A Streamlit dashboard that pulls recent video performance data (views, likes,
comments, engagement rate) for any public YouTube channel using the YouTube
Data API v3, and visualizes trends.

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

4. In the sidebar, enter a channel URL, `@handle`, or channel name and click
   **Analyze Channel**.

## Project structure

```
youtube-analytics-dashboard/
├── app.py            # Streamlit UI + charts
├── youtube_api.py     # YouTube Data API wrapper (channel resolution, video/stat fetching)
├── requirements.txt
├── .env.example
└── README.md
```

## What it does (v1)

- Resolves a channel from a URL, handle, or name
- Pulls channel-level stats (subs, total views, video count)
- Pulls the N most recent uploads and their view/like/comment counts
- Computes a simple engagement rate: `(likes + comments) / views * 100`
- Charts: views trend over time, top videos by views, engagement rate
  distribution, likes-vs-comments bubble chart
- Lets you export the pulled data as CSV

## Known limitations

- The view predictor only uses title/format features (length, word count,
  question/number/allcaps signals, duration, upload hour) — no thumbnail
  or topic/category data yet, so predictive power is limited. Model
  accuracy is reported via k-fold cross-validation (not a single
  train/test split) since sample sizes of 10-100 videos make a single
  split unstable; expect a wide `r2_std` on small samples, and treat
  predictions as directional signals rather than precise estimates.
- Engagement rate excludes videos where a creator has hidden the like
  count or disabled comments, rather than treating those as genuine
  zero-engagement videos. If every video on a channel has hidden likes,
  the "Avg. engagement" metric shows "N/A" instead of a number.
- Only a snapshot of the channel's most recent uploads is analyzed —
  there's no historical tracking of how a channel's stats change over time.

## Planned next steps

- Add a "compare two channels" mode
- Add thumbnail and topic/category features to the ML model
- Persist historical pulls so you can track growth over time, not just a
  snapshot of current stats

## Notes on API quota

Free tier = 10,000 units/day. Each `videos.list` / `channels.list` call is
~1 unit; `search.list` calls (used only as a fallback for channel resolution)
cost 100 units. Pulling 50 videos for a channel typically costs well under
50 units, so you have plenty of room to experiment.

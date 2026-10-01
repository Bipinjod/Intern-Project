"""
YouTube Analytics Dashboard
Prime Times Television — Junior Data Scientist Internship Project

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from googleapiclient.errors import HttpError

from config import load_config
from youtube_api import YouTubeAnalytics
from analytics import (
    add_engineered_features,
    flag_outliers,
    best_posting_times,
    train_view_predictor,
    predict_views,
)

try:
    SETTINGS = load_config()
except ValueError as exc:
    SETTINGS = None
    STARTUP_ERROR = str(exc)
else:
    STARTUP_ERROR = None

st.set_page_config(page_title="YouTube Analytics Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #0f172a;
            --panel: rgba(15, 23, 42, 0.72);
            --panel-soft: rgba(30, 41, 59, 0.84);
            --primary: #7c3aed;
            --secondary: #22c55e;
            --accent: #38bdf8;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.2);
        }

        .stApp {
            background: linear-gradient(135deg, #020817 0%, #0f172a 30%, #111827 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .dashboard-shell {
            background: rgba(15, 23, 42, 0.56);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.4rem 1.3rem 1.1rem;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.25);
            backdrop-filter: blur(10px);
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(56, 189, 248, 0.08));
            border: 1px solid rgba(124, 58, 237, 0.35);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.8));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem;
            margin-top: 0.5rem;
            height: 100%;
            min-height: 128px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }

        .metric-label {
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.6rem;
        }

        .metric-value {
            font-size: clamp(1.2rem, 2vw, 2rem);
            font-weight: 700;
            color: var(--text);
            line-height: 1.2;
        }

        .metric-delta {
            font-size: 0.8rem;
            color: #cbd5e1;
            margin-top: 0.4rem;
        }

        .section-wrap {
            background: rgba(15, 23, 42, 0.38);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.5rem;
            margin-top: 1rem;
        }

        .insight-pill {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.35);
            color: #bbf7d0;
            font-size: 0.78rem;
            font-weight: 600;
        }

        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.82);
            border-right: 1px solid var(--border);
        }

        .stTabs [role="tablist"] {
            gap: 0.5rem;
        }

        .stTabs [role="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.6rem 0.9rem;
            color: var(--text);
            border: 1px solid transparent;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: rgba(124, 58, 237, 0.18);
            border-color: rgba(124, 58, 237, 0.35);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, delta: str = "", accent: str = "primary"):
    accent_style = {
        "primary": "rgba(124, 58, 237, 0.18)",
        "green": "rgba(34, 197, 94, 0.12)",
        "blue": "rgba(56, 189, 248, 0.12)",
        "orange": "rgba(249, 115, 22, 0.12)",
    }.get(accent, "rgba(124, 58, 237, 0.18)")

    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card" style="background: linear-gradient(180deg, {accent_style}, rgba(15, 23, 42, 0.88));">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Cached fetch functions — avoids re-hitting the API (and burning quota)
# every time a widget triggers a rerun.
# ----------------------------------------------------------------------------
@st.cache_data(ttl=SETTINGS.cache_ttl if SETTINGS else 3600, show_spinner=False)
def fetch_channel_data(api_key: str, channel_input: str, max_videos: int):
    yt = YouTubeAnalytics(api_key)
    channel_id = yt.resolve_channel_id(channel_input)
    channel_info = yt.get_channel_info(channel_id)
    video_ids = yt.get_recent_video_ids(channel_info["uploads_playlist_id"], max_videos=max_videos)
    df = yt.get_video_stats(video_ids)
    return channel_info, df


# ----------------------------------------------------------------------------
# Sidebar - config
# ----------------------------------------------------------------------------
st.sidebar.title("📊 YouTube Analytics")
st.sidebar.markdown("Analyze any public channel and uncover what makes its content perform.")

if SETTINGS is None:
    st.error(STARTUP_ERROR)
    st.stop()

api_key = SETTINGS.api_key

channel_input = st.sidebar.text_input(
    "Channel URL, @handle, or name",
    placeholder="e.g. @MrBeast or https://youtube.com/@MrBeast",
)
max_videos = st.sidebar.slider("Number of recent videos to pull", 10, 100, SETTINGS.max_videos_default, step=10)
fetch_button = st.sidebar.button("Analyze Channel", type="primary")

st.sidebar.divider()
st.sidebar.subheader("Quick guide")
st.sidebar.caption("• Use a channel URL, handle, or name")
st.sidebar.caption("• Pull 10–100 recent videos for better pattern detection")
st.sidebar.caption("• Data is cached for 1 hour to protect quota")

st.sidebar.divider()
st.sidebar.caption("Prime Times Television internship project")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div>
                <div class="insight-pill">Dashboard</div>
                <h1 style="margin:0.5rem 0 0.2rem; font-size:2.2rem; color:#f8fafc;">YouTube Channel Analytics</h1>
                <p style="margin:0; color:#cbd5e1;">Track channel momentum, content performance, and the factors behind each upload.</p>
            </div>
            <div style="padding:0.5rem 0.8rem; border-radius:12px; background:rgba(15,23,42,0.5); border: 1px solid rgba(148,163,184,0.25); color:#e2e8f0; font-weight:600;">
                Live channel insights
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_key:
    st.warning(
        "No API key found. Create a `.env` file in this project folder "
        "(copy `.env.example` to `.env`) and paste your YouTube Data API v3 key in as "
        "`YOUTUBE_API_KEY=your_key_here`, then restart the app."
    )
    st.stop()

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.channel_info = None

if fetch_button:
    if not channel_input:
        st.error("Enter a channel URL, handle, or name first.")
    else:
        try:
            with st.spinner("Fetching and processing channel data..."):
                channel_info, df = fetch_channel_data(api_key, channel_input, max_videos)
                if not df.empty:
                    df = add_engineered_features(df)
                    df = flag_outliers(df)

            st.session_state.df = df
            st.session_state.channel_info = channel_info

        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status == 403:
                st.error(
                    "YouTube API quota exceeded or the API key is restricted. "
                    "Free-tier quota resets daily (Pacific time) — try again "
                    "later, or check the key's restrictions in Google Cloud Console."
                )
            elif status == 400:
                st.error("The YouTube API rejected the request — check that your API key is valid.")
            elif status == 404:
                st.error("Channel not found. Double-check the URL, handle, or name.")
            else:
                st.error(f"YouTube API error (status {status}): {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ----------------------------------------------------------------------------
# Display results
# ----------------------------------------------------------------------------
if st.session_state.channel_info:
    info = st.session_state.channel_info
    df = st.session_state.df

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    channel_row = st.columns([1, 5])
    with channel_row[0]:
        st.image(info["thumbnail"], width=120)
    with channel_row[1]:
        st.subheader(info["title"])
        description = info.get("description", "") or ""
        st.caption(description[:220] + ("..." if len(description) > 220 else ""))

    if df is not None and not df.empty:
        avg_engagement = df["engagement_rate"].mean()
        # Every video on some channels has hidden likes or disabled
        # comments (a real per-video creator setting) — in that case every
        # engagement_rate value is NaN and .mean() is also NaN, which would
        # otherwise render as the literal string "nan%" in the UI.
        avg_engagement_display = "N/A" if pd.isna(avg_engagement) else f"{avg_engagement:.2f}%"
        best_video = df.nlargest(1, "views").iloc[0]
        average_views = df["views"].mean()

        metric_cols = st.columns(4)
        with metric_cols[0]:
            metric_card("Subscribers", f"{info['subscriber_count']:,}", "Audience size", "primary")
        with metric_cols[1]:
            metric_card("Total views", f"{info['view_count']:,}", "Channel reach", "blue")
        with metric_cols[2]:
            metric_card("Uploads", f"{info['video_count']:,}", "Content library", "green")
        with metric_cols[3]:
            metric_card("Avg. engagement", avg_engagement_display, f"Top video: {best_video['views']:,} views", "orange")

        st.markdown(
            f"""
            <div style="margin: 1rem 0 0.35rem; display:flex; flex-wrap:wrap; gap:0.5rem;">
                <span class="insight-pill">Average views per video: {average_views:,.0f}</span>
                <span class="insight-pill">Best-performing upload: {best_video['title'][:50]}{'...' if len(best_video['title']) > 50 else ''}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No video data found for this channel.")

    st.markdown("</div>", unsafe_allow_html=True)

    if df is not None and not df.empty:
        tab_overview, tab_deep, tab_predict = st.tabs(
            ["📈 Overview", "🔬 Deep Analytics", "🎯 Performance Predictor"]
        )

        with tab_overview:
            st.subheader("Views Trend (Recent Videos)")
            fig_views = px.line(
                df.sort_values("published_at"),
                x="published_at",
                y="views",
                markers=True,
                hover_data=["title"],
                labels={"published_at": "Published Date", "views": "Views"},
                template="plotly_dark",
            )
            fig_views.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_views, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Videos by Views")
                top_views = df.nlargest(10, "views")[["title", "views", "likes", "comments"]]
                fig_bar = px.bar(top_views, x="views", y="title", orientation="h", template="plotly_dark")
                fig_bar.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                st.subheader("Engagement Rate Distribution")
                fig_hist = px.histogram(df, x="engagement_rate", nbins=20, template="plotly_dark")
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Video Data")
            display_df = df[["title", "published_at", "views", "likes", "comments", "engagement_rate", "views_per_day"]].copy()
            display_df["published_at"] = display_df["published_at"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download data as CSV", csv, "youtube_video_stats.csv", "text/csv")

        with tab_deep:
            st.subheader("🔥 Outlier Detection")
            outliers = df[df["is_viral_outlier"]]
            underperformers = df[df["is_underperformer"]]
            oc1, oc2 = st.columns(2)
            with oc1:
                st.markdown(f"**Viral outperformers ({len(outliers)})**")
                if not outliers.empty:
                    st.dataframe(
                        outliers[["title", "views", "view_zscore"]].sort_values("view_zscore", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("None detected in this sample.")
            with oc2:
                st.markdown(f"**Underperformers ({len(underperformers)})**")
                if not underperformers.empty:
                    st.dataframe(
                        underperformers[["title", "views", "view_zscore"]].sort_values("view_zscore"),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("None detected in this sample.")

            st.divider()

            st.subheader("📅 Best Day to Post")
            posting_stats = best_posting_times(df)
            if not posting_stats.empty:
                fig_days = px.bar(
                    posting_stats,
                    x="day_of_week",
                    y="avg_views",
                    hover_data=["video_count"],
                    labels={"day_of_week": "Day", "avg_views": "Avg Views"},
                    template="plotly_dark",
                )
                fig_days.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_days, use_container_width=True)
                best_day = posting_stats.loc[posting_stats["avg_views"].idxmax(), "day_of_week"]
                st.caption(f"Based on this sample, **{best_day}** uploads perform best on average.")

            st.divider()

            st.subheader("🎬 Format Analysis")
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**Duration vs Views**")
                fig_dur = px.scatter(
                    df,
                    x="duration_minutes",
                    y="views",
                    color="is_short",
                    hover_data=["title"],
                    labels={"duration_minutes": "Duration (min)", "is_short": "Is Short"},
                    template="plotly_dark",
                )
                fig_dur.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_dur, use_container_width=True)
            with fc2:
                st.markdown("**Title Length vs Views**")
                fig_title = px.scatter(
                    df,
                    x="title_length",
                    y="views",
                    hover_data=["title"],
                    labels={"title_length": "Title Length (chars)"},
                    template="plotly_dark",
                )
                fig_title.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_title, use_container_width=True)

            st.divider()
            st.subheader("📊 Correlation with Views")
            bool_cols = ["title_has_number", "title_has_question", "title_has_allcaps_word", "is_short"]
            corr_df = df.copy()
            for col in bool_cols:
                corr_df[col] = corr_df[col].astype(int)
            numeric_cols = [
                "title_length",
                "title_word_count",
                "duration_seconds",
                "hour_published",
                "likes",
                "comments",
                "engagement_rate",
                "views",
            ] + bool_cols
            corr = corr_df[numeric_cols].corr()["views"].drop("views").sort_values(ascending=False)
            fig_corr = px.bar(
                corr,
                orientation="h",
                labels={"value": "Correlation with Views", "index": "Feature"},
                template="plotly_dark",
            )
            fig_corr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(
                "Correlation ranges from -1 to 1. Values near 0 mean little linear "
                "relationship — doesn't rule out a non-linear effect, which is what "
                "the model in the Predictor tab is for."
            )

        with tab_predict:
            st.subheader("🎯 Predict Views for a New Video")
            st.caption(
                "Trains a Random Forest on this channel's title/format features "
                "to estimate expected views for a hypothetical upload. Needs at "
                "least 8 videos in the sample to train. Accuracy is evaluated with "
                "k-fold cross-validation rather than a single train/test split, "
                "since a single split is unreliable on samples this small."
            )

            if len(df) < 8:
                st.warning(f"Only {len(df)} videos loaded — pull at least 8 (ideally 30+) for a usable model.")
            else:
                try:
                    with st.spinner("Training model..."):
                        model, metrics, importance_df = train_view_predictor(df)

                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        metric_card(
                            "R² (cross-val avg)",
                            f"{metrics['r2']:.3f}",
                            f"± {metrics['r2_std']:.3f} across {metrics['n_folds']} folds",
                            "primary",
                        )
                    with mc2:
                        metric_card("Samples / Folds", f"{metrics['n_samples']} / {metrics['n_folds']}", "Data used", "blue")
                    with mc3:
                        metric_card("MAE (log)", f"{metrics['mae_log_scale']:.3f}", "Prediction error", "green")

                    if metrics["r2"] < 0.3:
                        st.caption(
                            "⚠️ Low R² — with a small sample, title/format features alone "
                            "don't explain much variance. This is a known limitation of "
                            "v1: no thumbnail or topic/category data is used yet. Treat "
                            "predictions as directional, not precise."
                        )

                    st.markdown("**Feature Importance**")
                    fig_imp = px.bar(
                        importance_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        template="plotly_dark",
                    )
                    fig_imp.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                    st.divider()
                    st.markdown("**Try it: estimate views for a hypothetical video**")
                    pred_title = st.text_input("Video title", value="How I Grew This Channel in 30 Days")
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        pred_duration_min = st.slider("Duration (minutes)", 0.5, 30.0, 8.0, step=0.5)
                    with pc2:
                        pred_hour = st.slider("Upload hour (24h, UTC)", 0, 23, 15)

                    if st.button("Predict views", type="primary"):
                        predicted = predict_views(
                            model,
                            pred_title,
                            int(pred_duration_min * 60),
                            pred_hour,
                        )
                        st.success(f"Estimated views: **{predicted:,}**")
                        st.caption(
                            "This is a rough estimate based on patterns in this channel's "
                            "own recent uploads — treat it as a directional signal, not a guarantee."
                        )

                except ValueError as e:
                    st.warning(str(e))
    else:
        st.info("No video data found for this channel.")
else:
    st.info("Enter a channel in the sidebar and click **Analyze Channel** to get started.")

st.markdown("</div>", unsafe_allow_html=True)
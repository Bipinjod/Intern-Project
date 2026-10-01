"""
youtube_api.py
--------------
Thin wrapper around the YouTube Data API v3.

Handles:
- Resolving a channel from a URL, @handle, or raw channel ID
- Fetching channel metadata (title, subs, total views, video count)
- Fetching the channel's uploaded videos (via the "uploads" playlist)
- Fetching per-video statistics (views, likes, comments) in batches

All functions raise exceptions on API errors so the Streamlit layer
can catch them and show a friendly message instead of crashing.
"""

import re
from typing import List, Dict

import numpy as np
import pandas as pd
from googleapiclient.discovery import build


class YouTubeAnalytics:
    def __init__(self, api_key: str, youtube_client=None):
        if not api_key:
            raise ValueError("YouTube API key is missing. Check your .env file.")
        self.youtube = youtube_client or build("youtube", "v3", developerKey=api_key)

    @staticmethod
    def _safe_int(value, default=0):
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    # Channel resolution
    # ------------------------------------------------------------------
    def resolve_channel_id(self, raw_input: str) -> str:
        """
        Accepts:
          - a raw channel ID (starts with 'UC...')
          - a full URL (youtube.com/channel/UC..., youtube.com/@handle, youtube.com/c/name)
          - a bare @handle
          - a plain channel name (falls back to search)
        Returns a resolved channel ID.
        """
        raw_input = raw_input.strip()

        # Already a channel ID
        if re.match(r"^UC[\w-]{22}$", raw_input):
            return raw_input

        # Extract from full URL
        url_match = re.search(r"youtube\.com/(channel/UC[\w-]{22}|@[\w.-]+|c/[\w.-]+)", raw_input)
        if url_match:
            path = url_match.group(1)
            if path.startswith("channel/"):
                return path.replace("channel/", "")
            handle_or_name = path.replace("@", "").replace("c/", "")
            return self._search_channel_id(handle_or_name)

        # Bare @handle
        if raw_input.startswith("@"):
            return self._search_channel_id(raw_input[1:])

        # Fallback: treat as search term (channel name)
        return self._search_channel_id(raw_input)

    def _search_channel_id(self, query: str) -> str:
        # Try the handle-based lookup first (cheap, exact)
        try:
            resp = self.youtube.channels().list(part="id", forHandle=query).execute()
            items = resp.get("items", [])
            if items:
                return items[0]["id"]
        except Exception:
            pass

        # Fallback to search.list (costs more quota but broader match)
        resp = self.youtube.search().list(
            part="snippet", q=query, type="channel", maxResults=1
        ).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError(f"No channel found for '{query}'")
        return items[0]["snippet"]["channelId"]

    # ------------------------------------------------------------------
    # Channel metadata
    # ------------------------------------------------------------------
    def get_channel_info(self, channel_id: str) -> Dict:
        resp = self.youtube.channels().list(
            part="snippet,statistics,contentDetails", id=channel_id
        ).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError("Channel not found.")
        item = items[0]
        snippet = item.get("snippet") or {}
        stats = item.get("statistics") or {}
        content_details = item.get("contentDetails") or {}
        related_playlists = content_details.get("relatedPlaylists") or {}
        uploads_playlist_id = related_playlists.get("uploads")
        if not uploads_playlist_id:
            raise ValueError("Channel uploads playlist not found.")
        return {
            "channel_id": channel_id,
            "title": snippet.get("title", "Unknown channel"),
            "description": snippet.get("description", ""),
            "thumbnail": ((snippet.get("thumbnails") or {}).get("high") or {}).get("url", ""),
            "subscriber_count": self._safe_int(stats.get("subscriberCount", 0)),
            "view_count": self._safe_int(stats.get("viewCount", 0)),
            "video_count": self._safe_int(stats.get("videoCount", 0)),
            "uploads_playlist_id": uploads_playlist_id,
        }

    # ------------------------------------------------------------------
    # Video listing + stats
    # ------------------------------------------------------------------
    def get_recent_video_ids(self, uploads_playlist_id: str, max_videos: int = 50) -> List[str]:
        video_ids = []
        next_page_token = None

        while len(video_ids) < max_videos:
            resp = self.youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=next_page_token,
            ).execute()

            video_ids.extend(
                item["contentDetails"]["videoId"] for item in resp.get("items", [])
            )
            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break

        return video_ids[:max_videos]

    def get_video_stats(self, video_ids: List[str]) -> pd.DataFrame:
        """Fetch stats for a list of video IDs, batching in chunks of 50 (API limit)."""
        rows = []

        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            resp = self.youtube.videos().list(
                part="snippet,statistics,contentDetails", id=",".join(batch)
            ).execute()

            for item in resp.get("items", []):
                if not isinstance(item, dict):
                    continue

                item_id = item.get("id")
                if not item_id:
                    continue

                stats = item.get("statistics") or {}
                snippet = item.get("snippet") or {}
                content_details = item.get("contentDetails") or {}

                title = snippet.get("title")
                published_at = snippet.get("publishedAt")
                duration = content_details.get("duration")
                if not title or not published_at or not duration:
                    continue

                # YouTube omits these keys entirely (rather than sending 0)
                # when a creator hides the like count or disables comments.
                likes_hidden = "likeCount" not in stats
                comments_disabled = "commentCount" not in stats
                rows.append({
                    "video_id": item_id,
                    "title": title,
                    "published_at": published_at,
                    "thumbnail": ((snippet.get("thumbnails") or {}).get("medium") or {}).get("url", ""),
                    "views": self._safe_int(stats.get("viewCount", 0)),
                    "likes": self._safe_int(stats.get("likeCount", 0)),
                    "comments": self._safe_int(stats.get("commentCount", 0)),
                    "likes_hidden": likes_hidden,
                    "comments_disabled": comments_disabled,
                    "duration": duration,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["published_at"] = pd.to_datetime(df["published_at"])
            df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
            # Simple engagement rate: (likes + comments) / views.
            # Use np.nan (not pd.NA) for the zero-views guard — pd.NA forces
            # the Series to object dtype, and object-dtype Series containing
            # pd.NA raise TypeError on .round() (NAType has no __round__).
            views_safe = df["views"].astype(float).replace(0, np.nan)
            engagement = (df["likes"] + df["comments"]) / views_safe * 100
            unmeasurable = df["likes_hidden"] | df["comments_disabled"]
            df["engagement_rate"] = engagement.where(~unmeasurable).round(2)
        return df

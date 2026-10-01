import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    app_mode: str = "development"
    cache_ttl: int = 3600
    max_videos_default: int = 30
    require_auth: bool = False


def load_config() -> AppConfig:
    app_mode = (os.getenv("APP_MODE") or "development").strip().lower()
    api_key = (os.getenv("YOUTUBE_API_KEY") or "").strip()
    cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
    max_videos_default = int(os.getenv("MAX_VIDEOS_DEFAULT", "30"))
    require_auth = _as_bool(os.getenv("REQUIRE_AUTH"), default=False)

    if app_mode == "production" and not api_key:
        raise ValueError("YOUTUBE_API_KEY is required in production mode.")

    return AppConfig(
        api_key=api_key,
        app_mode=app_mode,
        cache_ttl=max(60, cache_ttl),
        max_videos_default=max(10, max_videos_default),
        require_auth=require_auth,
    )

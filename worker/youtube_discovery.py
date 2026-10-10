"""Find YouTube videos for a text query.

Uses the YouTube Data API v3 when YOUTUBE_API_KEY is set, and yt-dlp search
otherwise. API failures fall through to yt-dlp so discovery still runs.
"""

import json
import urllib.parse
import urllib.request

import yt_dlp

from core.config import Config


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _normalize_entry(video_id: str, title: str = "", channel: str = "", url: str = "") -> dict:
    video_id = (video_id or "").strip()
    if not video_id:
        return None
    return {
        "video_id": video_id,
        "title": title or video_id,
        "channel": channel or "",
        "url": url or _watch_url(video_id),
    }


def _discover_with_data_api(query: str, limit: int) -> list:
    params = urllib.parse.urlencode({
        "part": "snippet",
        "type": "video",
        "maxResults": limit,
        "q": query,
        "key": Config.YOUTUBE_API_KEY,
        "safeSearch": "moderate",
        "videoEmbeddable": "true",
    })
    url = f"https://www.googleapis.com/youtube/v3/search?{params}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    videos = []
    for item in payload.get("items", []):
        id_block = item.get("id") or {}
        video_id = id_block.get("videoId")
        snippet = item.get("snippet") or {}
        normalized = _normalize_entry(
            video_id,
            title=snippet.get("title") or "",
            channel=snippet.get("channelTitle") or "",
        )
        if normalized:
            videos.append(normalized)
    return videos[:limit]


def _discover_with_ytdlp(query: str, limit: int) -> list:
    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(f"ytsearch{limit}:{query}", download=False)

    videos = []
    for entry in (info or {}).get("entries") or []:
        if not entry:
            continue
        video_id = entry.get("id") or ""
        url = entry.get("url") or entry.get("webpage_url") or ""
        if url and not url.startswith("http"):
            url = _watch_url(video_id)
        normalized = _normalize_entry(
            video_id,
            title=entry.get("title") or "",
            channel=entry.get("channel") or entry.get("uploader") or "",
            url=url,
        )
        if normalized:
            videos.append(normalized)
    return videos[:limit]


def discover_youtube_videos(query: str, limit: int = 5) -> list:
    """Return up to `limit` YouTube videos for a natural-language query."""
    query = (query or "").strip()
    if not query:
        return []
    limit = max(1, min(int(limit), 5))

    if Config.YOUTUBE_API_KEY:
        try:
            videos = _discover_with_data_api(query, limit)
            if videos:
                return videos
        except Exception as exc:
            print(f"[Discovery] YouTube Data API failed, falling back to yt-dlp: {exc}")

    return _discover_with_ytdlp(query, limit)

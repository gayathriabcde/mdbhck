import re
from urllib.parse import urlparse


_YOUTUBE_ID_RE = re.compile(r'(?:v=|vi=|youtu\.be/|embed/|shorts/|live/)([A-Za-z0-9_-]{11})')
_YOUTUBE_URL_RE = re.compile(r'https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+', re.IGNORECASE)


def is_valid_youtube_url(url: str) -> bool:
    """
    Validates if a given string is a properly formatted YouTube URL.
    Handles both standard (youtube.com) and short (youtu.be) formats.
    """
    if not url:
        return False

    parsed = urlparse(url)
    if parsed.netloc not in ['www.youtube.com', 'youtube.com', 'youtu.be', 'www.youtu.be', 'm.youtube.com']:
        return False

    return extract_youtube_video_id(url) is not None


def extract_youtube_video_id(url: str):
    """Return the 11-character video id from a URL or a bare id."""
    if not url:
        return None
    text = url.strip()
    match = _YOUTUBE_ID_RE.search(text)
    if match:
        return match.group(1)
    if re.fullmatch(r'[A-Za-z0-9_-]{11}', text):
        return text
    return None


def extract_youtube_urls(text: str) -> list:
    """Pull every YouTube URL out of a free-form paste."""
    if not text:
        return []
    urls = []
    seen = set()
    for match in _YOUTUBE_URL_RE.findall(text):
        cleaned = match.rstrip('.,);]')
        video_id = extract_youtube_video_id(cleaned)
        if video_id and video_id not in seen:
            seen.add(video_id)
            urls.append(cleaned)
    return urls


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def clamp_discover_limit(value, default=5) -> int:
    """Discovery returns the top 3 to 5 videos."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(3, min(5, parsed))


def citation_token(video_id: str, timestamp) -> str:
    """Stable token the UI turns into a clickable timestamp."""
    try:
        seconds = float(timestamp or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds.is_integer():
        rendered = str(int(seconds))
    else:
        rendered = f"{seconds:.1f}".rstrip("0").rstrip(".")
    return "{{cite:%s:%s}}" % (video_id, rendered)

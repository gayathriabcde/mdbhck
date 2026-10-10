import re
from urllib.parse import urlparse


def is_valid_youtube_url(url: str) -> bool:
    """
    Validates if a given string is a properly formatted YouTube URL.
    Handles both standard (youtube.com) and short (youtu.be) formats.
    """
    if not url:
        return False

    parsed = urlparse(url)
    if parsed.netloc not in ['www.youtube.com', 'youtube.com', 'youtu.be']:
        return False

    # Regex to ensure there is an actual video ID present
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )

    match = re.match(youtube_regex, url)
    return bool(match)
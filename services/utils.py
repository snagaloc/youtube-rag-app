import re
from urllib.parse import urlparse, parse_qs

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

def extract_video_id(url_or_id: str) -> str:
    """Accepts full YouTube URL or 11-char video ID."""
    s = (url_or_id or "").strip()

    if VIDEO_ID_RE.fullmatch(s):
        return s

    # youtu.be/<id>
    if "youtu.be" in s:
        path = urlparse(s).path.strip("/")
        if VIDEO_ID_RE.fullmatch(path):
            return path

    # youtube.com/watch?v=<id>
    parsed = urlparse(s)
    qs = parse_qs(parsed.query)
    vid = (qs.get("v", [""])[0] or "").strip()
    if VIDEO_ID_RE.fullmatch(vid):
        return vid

    # fallback regex
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)

    raise ValueError("Invalid YouTube URL/ID. Please paste a valid YouTube URL or 11-character video id.")

def sec_to_mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def youtube_timestamp_url(video_id: str, start_sec: float) -> str:
    t = max(0, int(start_sec))
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"
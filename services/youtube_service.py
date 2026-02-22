from dataclasses import dataclass
from typing import List, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

@dataclass
class TranscriptSnippet:
    text: str
    start: float
    duration: float

def _choose_transcript(tlist, preferred_lang: str = "en"):
    """
    Prefer:
      1) manually created in preferred_lang
      2) auto-generated in preferred_lang
      3) first available transcript
    """
    tr = None
    # These methods exist in many versions; wrap safely.
    try:
        tr = tlist.find_manually_created_transcript([preferred_lang])
    except Exception:
        pass
    if tr is None:
        try:
            tr = tlist.find_generated_transcript([preferred_lang])
        except Exception:
            pass
    if tr is None:
        tr = next(iter(tlist))
    return tr

def fetch_transcript(video_id: str, preferred_lang: str = "en") -> List[TranscriptSnippet]:
    """
    Fetch transcript with timestamps. Handles different youtube-transcript-api variants.
    Returns list of TranscriptSnippet.
    """
    try:
        api = YouTubeTranscriptApi()

        # Some installs expose classmethod list_transcripts; some expose instance .list()
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
        elif hasattr(api, "list"):
            tlist = api.list(video_id)
        else:
            raise RuntimeError("Unsupported youtube_transcript_api version (missing list/list_transcripts).")

        transcript = _choose_transcript(tlist, preferred_lang=preferred_lang)
        items = transcript.fetch()

        out: List[TranscriptSnippet] = []
        for it in items:
            # old versions return dicts, new versions return objects
            if isinstance(it, dict):
                out.append(TranscriptSnippet(
                    text=str(it.get("text", "")).strip(),
                    start=float(it.get("start", 0.0)),
                    duration=float(it.get("duration", 0.0)),
                ))
            else:
                out.append(TranscriptSnippet(
                    text=str(getattr(it, "text", "")).strip(),
                    start=float(getattr(it, "start", 0.0)),
                    duration=float(getattr(it, "duration", 0.0)),
                ))

        out = [x for x in out if x.text]
        if not out:
            raise RuntimeError("Transcript fetched but empty.")
        return out

    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript available for this video.")
"""Helpers for storing transcription payloads with optional word timings."""


def build_transcription_payload(segments=None, words=None):
    return {
        "segments": segments if isinstance(segments, list) else [],
        "words": words if isinstance(words, list) else [],
    }


def normalise_cached_transcription(data):
    """Handle both legacy ``[[text, start, end], ...]`` and dict payload caches."""
    if isinstance(data, dict):
        segments = data.get("segments", [])
        words = data.get("words", [])
        return (
            segments if isinstance(segments, list) else [],
            words if isinstance(words, list) else [],
        )

    return (data, []) if isinstance(data, list) else ([], [])

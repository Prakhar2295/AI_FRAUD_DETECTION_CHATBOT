"""Buffer for collecting streaming TTS chunks before playback."""

from __future__ import annotations


class ResponseBuffer:
    """A small buffer that accumulates PCM chunks for playback."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def append(self, chunk: bytes) -> None:
        self._chunks.append(chunk)

    def drain(self) -> bytes:
        audio = b"".join(self._chunks)
        self._chunks = []
        return audio

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def clear(self) -> None:
        self._chunks = []

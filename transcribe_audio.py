"""Async audio transcription via ElevenLabs Speech-to-Text (Scribe).

Usage:
    from transcribe_audio import transcribe_audio

    result = await transcribe_audio(audio_bytes, media_type="audio/webm")
    print(result["text"])

    result = await transcribe_audio(audio_bytes, media_type="audio/webm", diarize=True, timestamps="word")
    for word in result["words"]:
        print(f"[Speaker {word['speaker_id']}] {word['text']} ({word['start']}-{word['end']})")
"""

import asyncio
import os
import tempfile

from elevenlabs import ElevenLabs


def _get_client() -> ElevenLabs:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY environment variable is required for transcription")
    return ElevenLabs(api_key=api_key)


async def transcribe_audio(
    audio_bytes: bytes,
    *,
    media_type: str = "audio/webm",
    timestamps: str = "none",
    diarize: bool = False,
    num_speakers: int | None = None,
    language: str | None = None,
    model: str = "scribe_v2",
) -> dict:
    """Transcribe audio using ElevenLabs Speech-to-Text (Scribe v1 or v2)."""
    client = _get_client()

    # Pick file extension for content-type hint (SDK sends as multipart)
    ext = "webm"
    if "mp4" in media_type or "m4a" in media_type:
        ext = "m4a"
    elif "wav" in media_type:
        ext = "wav"
    elif "ogg" in media_type:
        ext = "ogg"
    elif "mpeg" in media_type or "mp3" in media_type:
        ext = "mp3"

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    def _convert():
        with open(tmp_path, "rb") as f:
            return client.speech_to_text.convert(
                file=f,
                model_id=model,
                language_code=language,
                diarize=diarize,
                num_speakers=num_speakers,
                timestamps_granularity=timestamps,
            )

    try:
        # ElevenLabs SDK is synchronous; run in thread pool to avoid blocking the event loop
        response = await asyncio.to_thread(_convert)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not response.text:
        raise RuntimeError("No transcription generated")

    words = []
    if getattr(response, "words", None):
        for w in response.words:
            words.append({
                "text": getattr(w, "text", str(w)),
                "start": getattr(w, "start", None),
                "end": getattr(w, "end", None),
                "speaker_id": getattr(w, "speaker_id", None),
            })

    return {
        "text": response.text,
        "language_code": getattr(response, "language_code", None),
        "words": words,
    }

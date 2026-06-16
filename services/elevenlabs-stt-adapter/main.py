"""
OpenAI-compatible STT adapter backed by ElevenLabs Scribe.

Vexa's transcription contract (stt.v1) is the OpenAI Audio API:
  POST /v1/audio/transcriptions  (multipart: file, model, response_format, language)
  -> { text, language, duration, segments:[{start,end,text,words:[{word,start,end,probability}]}] }

ElevenLabs Scribe speaks a different dialect (POST /v1/speech-to-text,
`xi-api-key` header, `model_id`, response with `words[]`/`logprob`/no segments).
This shim translates request + response so Vexa can use ElevenLabs unchanged.

Vexa already separates speakers (per-participant mono chunks), so we transcribe
each chunk and emit one segment — no diarization needed from ElevenLabs.

Env:
  ELEVENLABS_API_KEY   required — your ElevenLabs key (sent as xi-api-key)
  ELEVENLABS_MODEL     model_id (default "scribe_v1"; "scribe_v2" also valid)
  ELEVENLABS_STT_URL   override endpoint (default https://api.elevenlabs.io/v1/speech-to-text)
  ADAPTER_TOKEN        optional — if set, Vexa must send Authorization: Bearer <token>
  TAG_AUDIO_EVENTS     "false" (default) to keep (laughter)/(footsteps) out of transcripts
  PORT                 listen port (Railway injects)
"""
import logging
import math
import os
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("elevenlabs-stt-adapter")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # validated per-request so the service boots without it
ELEVENLABS_URL = os.getenv("ELEVENLABS_STT_URL", "https://api.elevenlabs.io/v1/speech-to-text")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "scribe_v1")
ADAPTER_TOKEN = os.getenv("ADAPTER_TOKEN", "")
TAG_AUDIO_EVENTS = os.getenv("TAG_AUDIO_EVENTS", "false")

app = FastAPI(title="ElevenLabs STT adapter (OpenAI-compatible)")


def map_to_verbose_json(el: dict) -> dict:
    """ElevenLabs SpeechToTextChunkResponseModel -> OpenAI verbose_json."""
    # Multichannel responses wrap per-channel transcripts; Vexa sends mono.
    if isinstance(el.get("transcripts"), list) and el["transcripts"]:
        el = el["transcripts"][0]

    text = el.get("text", "") or ""
    words: list[dict] = []
    logprobs: list[float] = []
    for w in el.get("words") or []:
        if w.get("type") != "word":  # skip "spacing" / "audio_event"
            continue
        lp = w.get("logprob")
        prob = max(0.0, min(1.0, math.exp(lp))) if isinstance(lp, (int, float)) else 1.0
        if isinstance(lp, (int, float)):
            logprobs.append(lp)
        words.append({
            "word": w.get("text", ""),
            "start": w.get("start") if w.get("start") is not None else 0.0,
            "end": w.get("end") if w.get("end") is not None else 0.0,
            "probability": prob,
        })

    duration = el.get("audio_duration_secs")
    if duration is None:
        duration = words[-1]["end"] if words else 0.0

    segments: list[dict] = []
    if text.strip():
        segments.append({
            "id": 0,
            "seek": 0,
            "start": words[0]["start"] if words else 0.0,
            "end": words[-1]["end"] if words else duration,
            "text": text,
            "avg_logprob": (sum(logprobs) / len(logprobs)) if logprobs else 0.0,
            "no_speech_prob": 0.0,
            "compression_ratio": 1.0,
            "words": words,
        })

    return {
        "text": text,
        "language": el.get("language_code", ""),
        "language_probability": el.get("language_probability"),
        "duration": duration,
        "segments": segments,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request, authorization: Optional[str] = Header(None)):
    if ADAPTER_TOKEN and authorization != f"Bearer {ADAPTER_TOKEN}":
        raise HTTPException(401, "invalid transcription token")
    if not ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY not configured")

    form = await request.form()
    upload: Any = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(400, "multipart field 'file' is required")
    audio = await upload.read()
    filename = getattr(upload, "filename", None) or "audio.wav"
    content_type = getattr(upload, "content_type", None) or "audio/wav"

    data = {
        "model_id": ELEVENLABS_MODEL,
        "timestamps_granularity": "word",
        "diarize": "false",
        "tag_audio_events": TAG_AUDIO_EVENTS,
        "file_format": "other",
    }
    language = form.get("language")
    if isinstance(language, str) and language and language.lower() not in ("auto", "none"):
        data["language_code"] = language

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            ELEVENLABS_URL,
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            data=data,
            files={"file": (filename, audio, content_type)},
        )
    if r.status_code != 200:
        log.warning("ElevenLabs STT %s: %s", r.status_code, r.text[:300])
        raise HTTPException(502, f"ElevenLabs STT error {r.status_code}: {r.text[:300]}")

    return map_to_verbose_json(r.json())

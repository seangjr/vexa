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
  ELEVENLABS_MODEL     model_id (default "scribe_v2"; "scribe_v1" also valid)
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

from singlish import KEYTERMS, REFINEMENT_SYSTEM_PROMPT, deterministic_normalize

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("elevenlabs-stt-adapter")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # validated per-request so the service boots without it
ELEVENLABS_URL = os.getenv("ELEVENLABS_STT_URL", "https://api.elevenlabs.io/v1/speech-to-text")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "scribe_v2")
ADAPTER_TOKEN = os.getenv("ADAPTER_TOKEN", "")
TAG_AUDIO_EVENTS = os.getenv("TAG_AUDIO_EVENTS", "false")

STT_KEYTERMS_ENABLED = os.getenv("STT_KEYTERMS_ENABLED", "true")
STT_EXTRA_KEYTERMS = os.getenv("STT_EXTRA_KEYTERMS", "")
STT_REFINEMENT_ENABLED = os.getenv("STT_REFINEMENT_ENABLED", "true")
STT_REFINEMENT_MODEL = os.getenv("STT_REFINEMENT_MODEL", "openai/gpt-4o-mini")
STT_REFINEMENT_API_KEY = os.getenv("STT_REFINEMENT_API_KEY", "")
STT_REFINEMENT_BASE_URL = os.getenv("STT_REFINEMENT_BASE_URL", "https://ai-gateway.vercel.sh/v1")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "")

app = FastAPI(title="ElevenLabs STT adapter (OpenAI-compatible)")


def _truthy(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


def _build_effective_keyterms() -> list[str]:
    seen: dict[str, None] = {}
    for term in KEYTERMS:
        if len(term) <= 50 and term not in seen:
            seen[term] = None
    if STT_EXTRA_KEYTERMS:
        for part in STT_EXTRA_KEYTERMS.split(","):
            term = part.strip()
            if term and len(term) <= 50 and term not in seen:
                seen[term] = None
    return list(seen)[:100]


_EFFECTIVE_KEYTERMS: list[str] = _build_effective_keyterms()


# ElevenLabs returns ISO-639-3 codes (e.g. "eng"); Vexa's TranscriptionSegment
# validates Whisper's ISO-639-1 set (e.g. "en"). Unmapped -> "en" so a segment
# always stores with a valid code instead of being dropped.
_VEXA_LANGS = {
    "af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy",
    "da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw",
    "he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn",
    "ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt",
    "my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si",
    "sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl",
    "tr","tt","uk","ur","uz","vi","yi","yo","yue","zh",
}
_ISO3_TO_1 = {
    "afr":"af","amh":"am","ara":"ar","asm":"as","aze":"az","bak":"ba","bel":"be",
    "bul":"bg","ben":"bn","bod":"bo","tib":"bo","bre":"br","bos":"bs","cat":"ca",
    "ces":"cs","cze":"cs","cym":"cy","wel":"cy","dan":"da","deu":"de","ger":"de",
    "ell":"el","gre":"el","eng":"en","spa":"es","est":"et","eus":"eu","baq":"eu",
    "fas":"fa","per":"fa","fin":"fi","fao":"fo","fra":"fr","fre":"fr","glg":"gl",
    "guj":"gu","hau":"ha","heb":"he","hin":"hi","hrv":"hr","hat":"ht","hun":"hu",
    "hye":"hy","arm":"hy","ind":"id","isl":"is","ice":"is","ita":"it","jpn":"ja",
    "jav":"jw","kat":"ka","geo":"ka","kaz":"kk","khm":"km","kan":"kn","kor":"ko",
    "lat":"la","ltz":"lb","lin":"ln","lao":"lo","lit":"lt","lav":"lv","mlg":"mg",
    "mri":"mi","mao":"mi","mkd":"mk","mac":"mk","mal":"ml","mon":"mn","mar":"mr",
    "msa":"ms","may":"ms","mlt":"mt","mya":"my","bur":"my","nep":"ne","nld":"nl",
    "dut":"nl","nno":"nn","nor":"no","nob":"no","oci":"oc","pan":"pa","pol":"pl",
    "pus":"ps","por":"pt","ron":"ro","rum":"ro","rus":"ru","san":"sa","snd":"sd",
    "sin":"si","slk":"sk","slo":"sk","slv":"sl","sna":"sn","som":"so","sqi":"sq",
    "alb":"sq","srp":"sr","sun":"su","swe":"sv","swa":"sw","tam":"ta","tel":"te",
    "tgk":"tg","tha":"th","tuk":"tk","tgl":"tl","tur":"tr","tat":"tt","ukr":"uk",
    "urd":"ur","uzb":"uz","vie":"vi","yid":"yi","yor":"yo","yue":"yue","zho":"zh",
    "chi":"zh","cmn":"zh",
}


def _normalize_lang(code) -> str:
    c = (code or "").strip().lower()
    if c in _VEXA_LANGS:
        return c
    return _ISO3_TO_1.get(c, "en")


def map_to_verbose_json(el: dict) -> dict:
    """ElevenLabs SpeechToTextChunkResponseModel -> OpenAI verbose_json."""
    # Multichannel responses wrap per-channel transcripts; Vexa sends mono.
    if isinstance(el.get("transcripts"), list) and el["transcripts"]:
        el = el["transcripts"][0]

    text = deterministic_normalize(el.get("text", "") or "")
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
        "language": _normalize_lang(el.get("language_code")),
        "language_probability": el.get("language_probability"),
        "duration": duration,
        "segments": segments,
    }


async def refine_text(client: httpx.AsyncClient, text: str) -> str:
    """LLM post-ASR correction. Strictly fail-open — never breaks transcription."""
    if not _truthy(STT_REFINEMENT_ENABLED) or not STT_REFINEMENT_API_KEY or not text.strip():
        return text
    try:
        r = await client.post(
            f"{STT_REFINEMENT_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {STT_REFINEMENT_API_KEY}"},
            json={
                "model": STT_REFINEMENT_MODEL,
                "temperature": 0,
                "max_tokens": max(64, min(1024, len(text) * 2)),
                "messages": [
                    {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            },
            timeout=10,
        )
        if r.status_code != 200:
            log.warning("LLM refinement %s: %s", r.status_code, r.text[:200])
            return text
        refined = r.json()["choices"][0]["message"]["content"].strip()
        return refined if refined else text
    except Exception as exc:
        log.warning("LLM refinement failed (non-fatal): %s", exc)
        return text


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

    # Language: STT_LANGUAGE env overrides form value; 'auto' -> omit language_code entirely.
    lang_override = STT_LANGUAGE.strip()
    if lang_override and lang_override.lower() == "auto":
        pass  # omit language_code entirely
    elif lang_override:
        data["language_code"] = lang_override
    else:
        language = form.get("language")
        if isinstance(language, str) and language and language.lower() not in ("auto", "none"):
            data["language_code"] = language

    # Keyterms biasing: scribe_v2 only.
    if (
        ELEVENLABS_MODEL.startswith("scribe_v2")
        and _truthy(STT_KEYTERMS_ENABLED)
        and _EFFECTIVE_KEYTERMS
    ):
        data["keyterms"] = _EFFECTIVE_KEYTERMS

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

        result = map_to_verbose_json(r.json())

        if result.get("text", "").strip():
            refined = await refine_text(client, result["text"])
            result["text"] = refined
            if result.get("segments"):
                result["segments"][0]["text"] = refined

    return result

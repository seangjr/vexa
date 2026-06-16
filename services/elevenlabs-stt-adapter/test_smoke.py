"""Offline smoke test for the ElevenLabs STT adapter.

Stubs the ElevenLabs + LLM-refinement HTTP calls (routed by URL). Live
verification happens after deploy. Run: python test_smoke.py
(deps: fastapi, httpx, python-multipart, pytest)
"""
import math
import os

os.environ.setdefault("ELEVENLABS_API_KEY", "xi-dummy")
os.environ.setdefault("ADAPTER_TOKEN", "shh")

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)

# A realistic ElevenLabs Scribe response (language as ISO-639-3 "eng").
EL_RESPONSE = {
    "language_code": "eng",
    "language_probability": 0.98,
    "text": "Hello world",
    "audio_duration_secs": 1.2,
    "words": [
        {"text": "Hello", "start": 0.0, "end": 0.5, "type": "word", "logprob": -0.01, "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.5, "end": 0.5, "type": "spacing", "logprob": 0.0},
        {"text": "world", "start": 0.5, "end": 1.0, "type": "word", "logprob": -0.2},
        {"text": "(laughter)", "start": 1.0, "end": 1.2, "type": "audio_event", "logprob": -0.5},
    ],
}


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _make_client(captured, refine_text="REFINED", refine_status=200, refine_raises=False):
    """Fake httpx.AsyncClient routing ElevenLabs vs the refinement gateway by URL."""
    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, data=None, files=None, json=None, timeout=None):
            if "chat/completions" in url:
                captured["refine"] = {"url": url, "headers": headers, "json": json}
                if refine_raises:
                    raise RuntimeError("boom")
                return _Resp(refine_status, {"choices": [{"message": {"content": refine_text}}]})
            captured["el"] = {"url": url, "headers": headers, "data": data}
            return _Resp(200, EL_RESPONSE)

    return _FakeClient


def _post(headers=None, data=None):
    files = {"file": ("chunk.wav", b"RIFFxxxxWAVE", "audio/wav")}
    return client.post(
        "/v1/audio/transcriptions",
        files=files,
        data=data or {"model": "large-v3-turbo", "language": "en"},
        headers=headers or {"Authorization": "Bearer shh"},
    )


def test_mapping_unit():
    out = main.map_to_verbose_json(EL_RESPONSE)
    assert out["text"] == "Hello world"
    assert out["language"] == "en"  # eng -> en
    assert out["duration"] == 1.2
    assert len(out["segments"]) == 1
    seg = out["segments"][0]
    assert [w["word"] for w in seg["words"]] == ["Hello", "world"]
    assert seg["start"] == 0.0 and seg["end"] == 1.0
    assert abs(seg["words"][0]["probability"] - math.exp(-0.01)) < 1e-9
    assert main.map_to_verbose_json({"transcripts": [EL_RESPONSE]})["text"] == "Hello world"
    assert main.map_to_verbose_json({"text": "", "words": []})["segments"] == []


def test_language_normalization():
    for src, want in [("eng", "en"), ("en", "en"), ("spa", "es"), ("jpn", "ja"),
                      ("zho", "zh"), ("haw", "haw"), ("yue", "yue"), ("ENG", "en"),
                      ("", "en"), ("xyz", "en")]:
        assert main._normalize_lang(src) == want


def test_glossary_contract():
    assert isinstance(main.KEYTERMS, list) and main.KEYTERMS
    assert len(main.KEYTERMS) <= 100
    assert all(isinstance(t, str) and 0 < len(t) <= 50 for t in main.KEYTERMS)
    assert len(main.KEYTERMS) == len(set(main.KEYTERMS))  # deduped
    assert "lah" in main.KEYTERMS
    assert isinstance(main.REFINEMENT_SYSTEM_PROMPT, str) and main.REFINEMENT_SYSTEM_PROMPT.strip()
    assert main.deterministic_normalize("hello world") == "hello world"
    assert main._EFFECTIVE_KEYTERMS and len(main._EFFECTIVE_KEYTERMS) <= 100


def test_auth_and_keyterms(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.httpx, "AsyncClient", _make_client(captured))
    monkeypatch.setattr(main, "STT_REFINEMENT_API_KEY", "")  # refinement off

    assert _post(headers={"Authorization": "Bearer nope"}).status_code == 401

    r = _post()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Hello world"
    assert body["segments"][0]["words"][0]["word"] == "Hello"
    assert captured["el"]["url"] == main.ELEVENLABS_URL
    assert captured["el"]["headers"]["xi-api-key"] == "xi-dummy"
    assert captured["el"]["data"]["model_id"] == "scribe_v2"
    assert captured["el"]["data"]["language_code"] == "en"
    assert captured["el"]["data"]["keyterms"] == main._EFFECTIVE_KEYTERMS  # biasing attached
    assert "refine" not in captured  # skipped without a key


def test_no_keyterms_for_scribe_v1(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.httpx, "AsyncClient", _make_client(captured))
    monkeypatch.setattr(main, "ELEVENLABS_MODEL", "scribe_v1")
    monkeypatch.setattr(main, "STT_REFINEMENT_API_KEY", "")
    assert _post().status_code == 200
    assert captured["el"]["data"]["model_id"] == "scribe_v1"
    assert "keyterms" not in captured["el"]["data"]


def test_refinement_replaces_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.httpx, "AsyncClient", _make_client(captured, refine_text="Hello world lah"))
    monkeypatch.setattr(main, "STT_REFINEMENT_API_KEY", "vck_test")
    r = _post()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Hello world lah"
    assert body["segments"][0]["text"] == "Hello world lah"
    assert "chat/completions" in captured["refine"]["url"]
    assert captured["refine"]["headers"]["Authorization"] == "Bearer vck_test"
    assert captured["refine"]["json"]["messages"][0]["content"] == main.REFINEMENT_SYSTEM_PROMPT


def test_refinement_fail_open_on_500(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.httpx, "AsyncClient", _make_client(captured, refine_status=500))
    monkeypatch.setattr(main, "STT_REFINEMENT_API_KEY", "vck_test")
    assert _post().json()["text"] == "Hello world"  # falls back to raw


def test_refinement_fail_open_on_exception(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.httpx, "AsyncClient", _make_client(captured, refine_raises=True))
    monkeypatch.setattr(main, "STT_REFINEMENT_API_KEY", "vck_test")
    assert _post().json()["text"] == "Hello world"


def test_language_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.httpx, "AsyncClient", _make_client(captured))
    monkeypatch.setattr(main, "STT_REFINEMENT_API_KEY", "")
    monkeypatch.setattr(main, "STT_LANGUAGE", "ms")
    _post(data={"model": "x", "language": "en"})
    assert captured["el"]["data"]["language_code"] == "ms"  # override wins
    captured.clear()
    monkeypatch.setattr(main, "STT_LANGUAGE", "auto")
    _post(data={"model": "x", "language": "en"})
    assert "language_code" not in captured["el"]["data"]  # auto -> omit


def test_missing_key_returns_503(monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", None)
    assert _post().status_code == 503


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

"""Offline smoke test: verifies request handling + ElevenLabs->verbose_json mapping
with the ElevenLabs HTTP call stubbed. Live verification happens once the real
ELEVENLABS_API_KEY is set and the adapter is deployed.

Run: python test_smoke.py  (deps: fastapi, httpx, python-multipart)
"""
import math
import os

os.environ.setdefault("ELEVENLABS_API_KEY", "xi-dummy")
os.environ.setdefault("ADAPTER_TOKEN", "shh")

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)

# A realistic ElevenLabs Scribe response.
EL_RESPONSE = {
    "language_code": "en",
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


def test_mapping_unit():
    out = main.map_to_verbose_json(EL_RESPONSE)
    assert out["text"] == "Hello world"
    assert out["language"] == "en"
    assert out["duration"] == 1.2
    assert len(out["segments"]) == 1
    seg = out["segments"][0]
    # only type=="word" entries survive (spacing + audio_event dropped)
    assert [w["word"] for w in seg["words"]] == ["Hello", "world"]
    assert seg["start"] == 0.0 and seg["end"] == 1.0
    assert abs(seg["words"][0]["probability"] - math.exp(-0.01)) < 1e-9
    # multichannel wrapper is unwrapped
    assert main.map_to_verbose_json({"transcripts": [EL_RESPONSE]})["text"] == "Hello world"
    # silence -> no segments
    assert main.map_to_verbose_json({"text": "", "words": []})["segments"] == []


def test_endpoint_auth_and_flow(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return EL_RESPONSE

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, data=None, files=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["data"] = data
            return _Resp()

    monkeypatch.setattr(main.httpx, "AsyncClient", _FakeClient)

    files = {"file": ("chunk.wav", b"RIFFxxxxWAVE", "audio/wav")}
    data = {"model": "large-v3-turbo", "language": "en", "response_format": "verbose_json"}

    # wrong token -> 401
    r = client.post("/v1/audio/transcriptions", files=files, data=data, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401, r.text

    # correct token -> mapped verbose_json
    r = client.post("/v1/audio/transcriptions", files=files, data=data, headers={"Authorization": "Bearer shh"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Hello world" and body["segments"][0]["words"][0]["word"] == "Hello"
    # adapter called ElevenLabs correctly
    assert captured["url"] == main.ELEVENLABS_URL
    assert captured["headers"]["xi-api-key"] == "xi-dummy"
    assert captured["data"]["model_id"] == "scribe_v1"
    assert captured["data"]["language_code"] == "en"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

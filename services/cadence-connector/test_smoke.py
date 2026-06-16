"""Offline smoke test: exercises the webhook logic with stubbed network calls.

Run: python test_smoke.py  (deps: fastapi, httpx)
End-to-end (real Vexa + Twenty) verification happens after the Cadence API key
is provided and the service is deployed.
"""
import os

os.environ.setdefault("VEXA_API_URL", "https://vexa.example")
os.environ.setdefault("VEXA_API_TOKEN", "tok")
os.environ.setdefault("TWENTY_API_URL", "https://crm.example")
os.environ.setdefault("TWENTY_API_KEY", "key")
os.environ.setdefault("VEXA_WEBHOOK_SECRET", "shh")

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_build_note():
    note = main.build_note(
        {"platform": "google_meet", "native_meeting_id": "abc", "constructed_meeting_url": "https://m/abc"},
        {"segments": [{"speaker": "Alice", "text": "Hi"}, {"speaker": "Bob", "text": "Yo"}]},
    )
    assert "Alice" in note["body"] and "Hi" in note["body"]
    assert "google_meet" in note["title"]


def test_extract_helpers():
    assert main._extract_id({"data": {"createNote": {"id": "n1"}}}) == "n1"
    assert main._extract_id({"data": {"id": "n2"}}) == "n2"
    assert main._extract_list({"data": [{"id": "p1"}]})[0]["id"] == "p1"
    assert main._emails({"data": {"attendees": [{"email": "a@b.com"}]}}, {}) == ["a@b.com"]
    assert main._segments({"segments": [{"text": "x"}]})[0]["text"] == "x"


def _hook(payload, auth="Bearer shh"):
    return client.post("/vexa/webhook", json=payload, headers=({"Authorization": auth} if auth else {}))


def test_webhook_flow():
    async def _fetch(_c, _p, _n):
        return {"segments": [{"speaker": "Alice", "text": "Hello world"}]}

    async def _create(_c, _t, _b):
        return "note-123"

    async def _link(_c, _nid, _m, _t):
        return {"email": "a@b.com", "person_id": "p1"}

    main.fetch_transcript = _fetch
    main.create_twenty_note = _create
    main.maybe_link_person = _link
    main._seen.clear()

    done = {"event_type": "meeting.status_change",
            "meeting": {"status": "completed", "platform": "google_meet", "native_meeting_id": "abc"}}

    assert _hook(done, auth="Bearer wrong").status_code == 401
    assert _hook({"event_type": "meeting.status_change",
                  "meeting": {"status": "active", "platform": "google_meet", "native_meeting_id": "abc"}}
                 ).json().get("ignored") == "meeting.status_change"
    j = _hook(done).json()
    assert j.get("ok") and j.get("note_id") == "note-123" and j.get("linked"), j
    assert _hook(done).json().get("deduped"), "expected dedup on repeat"


if __name__ == "__main__":
    test_health()
    test_build_note()
    test_extract_helpers()
    test_webhook_flow()
    print("ALL OK")

"""
Vexa -> Cadence (Twenty CRM) connector.

Receives Vexa meeting webhooks, pulls the finished transcript from the Vexa
API, and writes it into Cadence (Twenty CRM) as a Note — best-effort linked to
a Person matched by attendee email.

Direction: Vexa -> CRM (push). Single, stateless HTTP service.

Env:
  VEXA_API_URL          Vexa API gateway base, e.g. https://vexa-...up.railway.app
  VEXA_API_TOKEN        Vexa user API token with `tx` scope (reads transcripts)
  VEXA_WEBHOOK_SECRET   Shared secret; Vexa sends `Authorization: Bearer <secret>`
  TWENTY_API_URL        Cadence base, e.g. https://crm.khaeli.com
  TWENTY_API_KEY        Cadence (Twenty) API key (Settings -> APIs & Webhooks)
  MATCH_BY_EMAIL        "true" (default) to link the note to a Person by email
  PORT                  Listen port (Railway injects this)
"""
import hmac
import logging
import os
import time
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger("cadence-connector")


def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"{name} is required")
    return v.rstrip("/") if name.endswith("_URL") else v


VEXA_API_URL = _require("VEXA_API_URL")
VEXA_API_TOKEN = _require("VEXA_API_TOKEN")
VEXA_WEBHOOK_SECRET = os.getenv("VEXA_WEBHOOK_SECRET", "")
TWENTY_API_URL = _require("TWENTY_API_URL")
TWENTY_API_KEY = _require("TWENTY_API_KEY")
MATCH_BY_EMAIL = os.getenv("MATCH_BY_EMAIL", "true").lower() in ("1", "true", "yes")

app = FastAPI(title="Vexa -> Cadence connector")

# Best-effort idempotency: webhooks are at-least-once. Single instance, so an
# in-memory TTL cache keeps retries from creating duplicate notes within the
# window. (Across restarts a duplicate is possible; Twenty has no natural
# upsert key for free-form notes.)
_seen: dict[str, float] = {}
_DEDUP_TTL = 6 * 3600


def _twenty_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TWENTY_API_KEY}", "Content-Type": "application/json"}


def _extract_id(data: Any) -> Optional[str]:
    """Twenty REST create responses vary by version; dig out the record id."""
    if not isinstance(data, dict):
        return None
    node = data.get("data", data)
    if isinstance(node, dict):
        # {"data": {"createNote": {"id": ...}}} or {"data": {"id": ...}}
        for v in (node.get("createNote"), node):
            if isinstance(v, dict) and v.get("id"):
                return v["id"]
    return None


def _extract_list(data: Any) -> list:
    """Pull a record list out of a Twenty REST list response."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        node = data.get("data", data)
        if isinstance(node, list):
            return node
        if isinstance(node, dict):
            for key in ("people", "records", "edges"):
                v = node.get(key)
                if isinstance(v, list):
                    return [e.get("node", e) if isinstance(e, dict) else e for e in v]
    return []


def _segments(transcript: Any) -> list[dict]:
    if isinstance(transcript, dict):
        for key in ("segments", "transcripts", "transcript", "items"):
            v = transcript.get(key)
            if isinstance(v, list):
                return v
    return transcript if isinstance(transcript, list) else []


def build_note(meeting: dict, transcript: Any) -> dict[str, str]:
    platform = meeting.get("platform", "meeting")
    native_id = meeting.get("native_meeting_id", "")
    lines = []
    for seg in _segments(transcript):
        if not isinstance(seg, dict):
            continue
        speaker = seg.get("speaker") or seg.get("speaker_name") or "Speaker"
        text = (seg.get("text") or seg.get("content") or "").strip()
        if text:
            lines.append(f"**{speaker}:** {text}")
    body_text = "\n\n".join(lines) if lines else "_No transcript text available._"
    url = meeting.get("constructed_meeting_url") or ""
    ended = meeting.get("end_time") or ""
    meta = [f"Platform: {platform}", f"Meeting: {native_id}"]
    if url:
        meta.append(f"URL: {url}")
    if ended:
        meta.append(f"Ended: {ended}")
    body = "  \n".join(meta) + "\n\n---\n\n" + body_text
    return {"title": f"Meeting transcript — {platform} {native_id}".strip(), "body": body}


async def fetch_transcript(client: httpx.AsyncClient, platform: str, native_id: str) -> Any:
    r = await client.get(
        f"{VEXA_API_URL}/transcripts/{platform}/{native_id}",
        headers={"X-API-Key": VEXA_API_TOKEN},
    )
    r.raise_for_status()
    return r.json()


async def create_twenty_note(client: httpx.AsyncClient, title: str, body: str) -> Optional[str]:
    # Newer Twenty uses bodyV2 (rich text). Older uses body (plain). Try modern
    # first, fall back on 400 (schema mismatch) so we work across fork versions.
    last: Optional[httpx.Response] = None
    for payload in ({"title": title, "bodyV2": {"markdown": body}}, {"title": title, "body": body}):
        r = await client.post(f"{TWENTY_API_URL}/rest/notes", headers=_twenty_headers(), json=payload)
        last = r
        if r.status_code in (200, 201):
            return _extract_id(r.json())
        if r.status_code != 400:
            r.raise_for_status()
    raise HTTPException(502, f"Twenty note create failed: {last.status_code if last else '?'} {last.text if last else ''}")


def _emails(meeting: dict, transcript: Any) -> list[str]:
    found: list[str] = []
    data = meeting.get("data") or {}
    for att in (data.get("attendees") or data.get("participants") or []):
        email = att.get("email") if isinstance(att, dict) else (att if isinstance(att, str) else None)
        if email and "@" in email and email not in found:
            found.append(email)
    if isinstance(transcript, dict):
        for p in (transcript.get("participants") or []):
            email = p.get("email") if isinstance(p, dict) else None
            if email and "@" in email and email not in found:
                found.append(email)
    return found


async def maybe_link_person(client: httpx.AsyncClient, note_id: str, meeting: dict, transcript: Any) -> Optional[dict]:
    for email in _emails(meeting, transcript):
        r = await client.get(
            f"{TWENTY_API_URL}/rest/people",
            headers=_twenty_headers(),
            params={"filter": f"emails.primaryEmail[eq]:{email}"},
        )
        if r.status_code != 200:
            continue
        people = _extract_list(r.json())
        if not people:
            continue
        pid = people[0].get("id")
        if not pid:
            continue
        lr = await client.post(
            f"{TWENTY_API_URL}/rest/noteTargets",
            headers=_twenty_headers(),
            json={"noteId": note_id, "personId": pid},
        )
        if lr.status_code in (200, 201):
            return {"email": email, "person_id": pid}
    return None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/vexa/webhook")
async def vexa_webhook(request: Request, authorization: Optional[str] = Header(None)):
    if VEXA_WEBHOOK_SECRET:
        expected = f"Bearer {VEXA_WEBHOOK_SECRET}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(401, "invalid webhook secret")

    payload = await request.json()
    event_type = payload.get("event_type")
    meeting = payload.get("meeting") or {}
    status = meeting.get("status")
    to_status = (payload.get("status_change") or {}).get("to")

    completed = event_type == "recording.completed" or (
        event_type == "meeting.status_change" and "completed" in (status, to_status)
    )
    if not completed:
        return {"ignored": event_type, "status": status}

    platform = meeting.get("platform")
    native_id = meeting.get("native_meeting_id")
    if not platform or not native_id:
        raise HTTPException(400, "missing meeting.platform / meeting.native_meeting_id")

    key = f"{platform}/{native_id}"
    now = time.time()
    if key in _seen and now - _seen[key] < _DEDUP_TTL:
        return {"deduped": key}

    async with httpx.AsyncClient(timeout=30.0) as client:
        transcript = await fetch_transcript(client, platform, native_id)
        note = build_note(meeting, transcript)
        note_id = await create_twenty_note(client, note["title"], note["body"])
        linked = await maybe_link_person(client, note_id, meeting, transcript) if (MATCH_BY_EMAIL and note_id) else None

    _seen[key] = now
    log.info("pushed meeting %s -> Twenty note %s (linked=%s)", key, note_id, linked)
    return {"ok": True, "meeting": key, "note_id": note_id, "linked": linked}

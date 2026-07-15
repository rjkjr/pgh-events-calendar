#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import time

from dotenv import load_dotenv
import anthropic
import httpx

REPO_DIR = Path(__file__).resolve().parent
load_dotenv(REPO_DIR / ".env")

with open(REPO_DIR / "config.json") as f:
    CONFIG = json.load(f)

TZ = ZoneInfo(CONFIG["timezone"])
MODEL = CONFIG.get("model", "claude-sonnet-5")

# Extra retries so a transient 429/529/5xx (e.g. API overloaded) doesn't abort
# the whole refresh — the SDK retries these with backoff. Keep the SDK's default
# 10-minute timeout; heavy web-search turns are streamed (below) to avoid
# idle-connection read timeouts.
client = anthropic.Anthropic(max_retries=6)


def research_events(query: str) -> str:
    today = datetime.now(TZ)
    system = (
        "You are a local events researcher for Pittsburgh, PA. Use web search "
        "to find real, currently-scheduled events from the top search results. "
        "Only include events you found actual evidence for — do not invent events. "
        f"Today's real date is {today.strftime('%A, %B %d, %Y')}. Search results "
        "often surface pages from a prior year's recurring event — always resolve "
        "each event to its next actual occurrence on or after today's date, and "
        "report the correct year explicitly. Discard any event whose confirmed "
        "date has already passed relative to today."
    )
    messages = [{
        "role": "user",
        "content": (
            f"Search for \"{query}\" and compile a thorough, well-organized list "
            f"of up to {CONFIG['max_events']} events happening in Pittsburgh, PA "
            "for this period, starting from today. For each event, report on its "
            "own labeled lines:\n"
            "- Event name\n"
            "- Date(s) with the correct year\n"
            "- Start time (and end time) if known\n"
            "- Venue / location (be as specific as the source allows)\n"
            "- One-sentence description\n"
            "- Source URL: the exact web address of the page you found this "
            "event on (an event detail page or venue calendar page — prefer a "
            "direct link over a generic homepage). Copy the full https:// URL "
            "verbatim from the search results; do not invent, shorten, or guess "
            "URLs. If you genuinely have no URL for an event, write 'Source URL: "
            "none'."
        ),
    }]

    # The server-side web_search loop can hit its iteration limit and return
    # stop_reason "pause_turn" with only a partial (preamble) response. When
    # that happens we must append the assistant turn and re-send so the server
    # resumes and finishes compiling the list — otherwise research comes back
    # nearly empty and zero events are parsed.
    text_parts: list[str] = []
    for _ in range(6):  # cap pause_turn resumes to avoid an infinite loop
        # Stream the request: web-search turns can run long, and a non-streaming
        # request holds an idle connection that trips the HTTP read timeout.
        # A long stream can also drop mid-body (RemoteProtocolError) or the API
        # can be transiently overloaded (529) — the SDK's own retry doesn't
        # cover a stream that dies after bytes start flowing, so retry here.
        response = _stream_with_retry(system, messages)
        text_parts.extend(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": response.content})

    return "\n".join(t for t in text_parts if t)


def _stream_with_retry(system: str, messages: list, attempts: int = 5):
    """Run one streamed research turn, retrying transient network/API errors."""
    transient = (
        anthropic.APIStatusError,      # 429 / 5xx (incl. 529 overloaded)
        anthropic.APIConnectionError,  # timeouts, dropped connections
        httpx.HTTPError,               # mid-stream RemoteProtocolError, etc.
    )
    for attempt in range(attempts):
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=8192,
                output_config={"effort": "medium"},
                tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}],
                system=system,
                messages=messages,
            ) as stream:
                return stream.get_final_message()
        except transient as e:
            if attempt == attempts - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(f"Transient API error ({type(e).__name__}); retrying in {wait}s "
                  f"[{attempt + 1}/{attempts - 1}]", file=sys.stderr)
            time.sleep(wait)


def extract_structured_events(research_text: str) -> list[dict]:
    schema_hint = (
        '[{"title": str, "start_date": "YYYY-MM-DD", "start_time": "HH:MM" or null, '
        '"end_date": "YYYY-MM-DD" or null, "end_time": "HH:MM" or null, '
        '"all_day": bool, "location": str, "description": str, "url": str or null}]'
    )
    # 16000 tokens leaves ample room for a full month of events — 4096 was too
    # small and truncated the JSON mid-object. Stream to stay under HTTP timeouts.
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        messages=[{
            "role": "user",
            "content": (
                "Convert the following research notes into a JSON array matching "
                f"exactly this shape (no markdown fences, no prose, output ONLY the "
                f"JSON array):\n{schema_hint}\n\n"
                "Times are 24-hour local Pittsburgh time (America/New_York). If a "
                "date is ambiguous or missing, omit that event. Today's real date is "
                f"{datetime.now(TZ).strftime('%Y-%m-%d')} — every start_date/end_date "
                "must be on or after today; if the notes give a date before today, "
                "correct it to the next occurrence of that date.\n"
                "For 'url': copy the event's Source URL verbatim from the notes when "
                "one is given; use null only when the notes say 'none' or give no URL. "
                "Never fabricate or guess a URL.\n"
                "For 'location': copy the venue/location text from the notes; use an "
                "empty string only if the notes give no location.\n"
                "Research notes:\n\n"
                f"{research_text}"
            ),
        }],
    ) as stream:
        response = stream.get_final_message()
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse structured events JSON: {e}", file=sys.stderr)
        salvaged = _salvage_json_objects(text)
        if salvaged:
            print(f"Salvaged {len(salvaged)} complete events from truncated JSON.",
                  file=sys.stderr)
            return salvaged
        print(text, file=sys.stderr)
        return []


def _salvage_json_objects(text: str) -> list[dict]:
    """Recover the complete top-level objects from a truncated JSON array.

    If the model's array is cut off mid-object, decode object-by-object and keep
    every one that parses cleanly, discarding only the final incomplete fragment.
    """
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break  # reached the truncated fragment
        if isinstance(obj, dict):
            objects.append(obj)
        idx = text.find("{", end)
    return objects


def escape_ics(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def build_ics(events: list[dict]) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//PGHWIDGET//Pittsburgh Events//EN",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{escape_ics(CONFIG['calendar_name'])}",
        "REFRESH-INTERVAL;VALUE=DURATION:PT6H",
        "X-PUBLISHED-TTL:PT6H",
    ]

    for ev in events:
        try:
            start_date = ev["start_date"]
            title = ev.get("title") or "Untitled Event"
        except KeyError:
            continue

        uid = uuid.uuid5(uuid.NAMESPACE_URL, f"{title}|{start_date}|{ev.get('location','')}")
        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{uid}@pghwidget")
        lines.append(f"DTSTAMP:{now_utc}")
        lines.append(f"SUMMARY:{escape_ics(title)}")

        start_time = ev.get("start_time")
        if ev.get("all_day") or not start_time:
            end_date = ev.get("end_date") or start_date
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            lines.append(f"DTSTART;VALUE=DATE:{start_date.replace('-', '')}")
            lines.append(f"DTEND;VALUE=DATE:{end_dt.strftime('%Y%m%d')}")
        else:
            # A timed event spanning multiple calendar days (e.g. "7-9PM, July 15
            # through July 18") means the same time window repeats each day, not
            # one continuous block from day-1-start to day-N-end. Model that as a
            # daily-recurring VEVENT anchored to start_date, not a multi-day span.
            start_local = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            end_time = ev.get("end_time")
            if end_time:
                end_local = datetime.strptime(f"{start_date} {end_time}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
                if end_local <= start_local:
                    end_local += timedelta(days=1)  # crosses midnight
            else:
                end_local = start_local + timedelta(hours=2)
            lines.append(f"DTSTART:{start_local.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            lines.append(f"DTEND:{end_local.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

            end_date = ev.get("end_date") or start_date
            if end_date != start_date:
                until_local = datetime.strptime(f"{end_date} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
                until_utc = until_local.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                lines.append(f"RRULE:FREQ=DAILY;UNTIL={until_utc}")

        location = ev.get("location") or ""
        description = ev.get("description") or ""

        # Only treat a value as a real URL if it's an http(s) link — guard against
        # the model emitting "none"/"" or a bare fragment.
        raw_url = (ev.get("url") or "").strip()
        url = raw_url if raw_url.lower().startswith(("http://", "https://")) else ""

        if url:
            description = f"{description}\\n\\n{url}"
        if location:
            lines.append(f"LOCATION:{escape_ics(location)}")
        lines.append(f"DESCRIPTION:{escape_ics(description)}")
        if url:
            lines.append(f"URL:{url}")
        lines.append("END:VEVENT")

    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def git_publish():
    subprocess.run(["git", "add", "events.ics"], cwd=REPO_DIR, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_DIR)
    if diff.returncode == 0:
        print("No changes to events.ics — skipping commit.")
        return
    subprocess.run(
        ["git", "commit", "-m", f"Update events feed ({datetime.now(TZ).isoformat()})"],
        cwd=REPO_DIR, check=True,
    )
    subprocess.run(["git", "push"], cwd=REPO_DIR, check=True)
    print("Pushed updated events.ics.")


def main():
    query = CONFIG["query_template"].format(period=CONFIG["period"])
    print(f"Researching: {query}")
    research_text = research_events(query)
    events = extract_structured_events(research_text)
    print(f"Parsed {len(events)} events.")

    # Fail-safe: never overwrite a good published feed with an empty one. A
    # transient research/extraction failure (e.g. a paused web-search turn)
    # would otherwise blank out the calendar. Keep the last good events.ics.
    ics_path = REPO_DIR / "events.ics"
    if not events:
        existing = ics_path.read_text() if ics_path.exists() else ""
        if "BEGIN:VEVENT" in existing:
            print("Parsed 0 events — keeping existing feed (not overwriting). "
                  "This is usually a transient search hiccup; try again.")
            return
        print("Parsed 0 events and no existing feed to preserve — writing empty calendar.")

    ics_content = build_ics(events)
    ics_path.write_text(ics_content)
    git_publish()


if __name__ == "__main__":
    main()

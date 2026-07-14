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

from dotenv import load_dotenv
import anthropic

REPO_DIR = Path(__file__).resolve().parent
load_dotenv(REPO_DIR / ".env")

with open(REPO_DIR / "config.json") as f:
    CONFIG = json.load(f)

TZ = ZoneInfo(CONFIG["timezone"])
MODEL = CONFIG.get("model", "claude-sonnet-5")

client = anthropic.Anthropic()


def research_events(query: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"effort": "medium"},
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 6}],
        system=(
            "You are a local events researcher for Pittsburgh, PA. Use web search "
            "to find real, currently-scheduled events from the top search results. "
            "Only include events you found actual evidence for — do not invent events."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Search for \"{query}\" and compile a thorough, well-organized list "
                f"of up to {CONFIG['max_events']} events happening in Pittsburgh, PA "
                "for this period. For each event, report: the event name, date(s), "
                "start time if known, venue or location, a one-sentence description, "
                "and the source URL if available."
            ),
        }],
    )
    return "\n".join(b.text for b in response.content if b.type == "text")


def extract_structured_events(research_text: str) -> list[dict]:
    schema_hint = (
        '[{"title": str, "start_date": "YYYY-MM-DD", "start_time": "HH:MM" or null, '
        '"end_date": "YYYY-MM-DD" or null, "end_time": "HH:MM" or null, '
        '"all_day": bool, "location": str, "description": str, "url": str or null}]'
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"effort": "low"},
        messages=[{
            "role": "user",
            "content": (
                "Convert the following research notes into a JSON array matching "
                f"exactly this shape (no markdown fences, no prose, output ONLY the "
                f"JSON array):\n{schema_hint}\n\n"
                "Times are 24-hour local Pittsburgh time (America/New_York). If a "
                "date is ambiguous or missing, omit that event. Research notes:\n\n"
                f"{research_text}"
            ),
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse structured events JSON: {e}", file=sys.stderr)
        print(text, file=sys.stderr)
        return []


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
            start_local = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            end_date = ev.get("end_date") or start_date
            end_time = ev.get("end_time")
            if end_time:
                end_local = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
            else:
                end_local = start_local + timedelta(hours=2)
            lines.append(f"DTSTART:{start_local.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            lines.append(f"DTEND:{end_local.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

        location = ev.get("location") or ""
        description = ev.get("description") or ""
        url = ev.get("url")
        if url:
            description = f"{description}\\n\\n{url}"
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
    ics_content = build_ics(events)
    (REPO_DIR / "events.ics").write_text(ics_content)
    git_publish()


if __name__ == "__main__":
    main()

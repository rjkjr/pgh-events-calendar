#!/usr/bin/env python3
"""Small desktop control panel for the Pittsburgh Events calendar feed.

Edit the search query, period, cadence, and other basic inputs, save them to
config.json, and manually trigger a refresh (which runs generate_events.py,
regenerates events.ics, and pushes it to GitHub).
"""
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

REPO_DIR = Path(__file__).resolve().parent
CONFIG_PATH = REPO_DIR / "config.json"
FEED_URL = "https://raw.githubusercontent.com/rjkjr/pgh-events-calendar/main/events.ics"

# (config key, label, widget kind). "text" = free text, "int" = integer entry.
FIELDS = [
    ("query_template", "Search query template", "text"),
    ("period", "Period (e.g. week, weekend, month)", "text"),
    ("cadence_hours", "Auto-refresh cadence (hours)", "int"),
    ("max_events", "Max events", "int"),
    ("calendar_name", "Calendar name", "text"),
    ("timezone", "Timezone", "text"),
    ("model", "Model", "text"),
]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


class ControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pittsburgh Events — Control Panel")
        self.geometry("640x620")
        self.minsize(560, 560)

        self.config_data = load_config()
        self.vars: dict[str, tk.StringVar] = {}
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.refresh_running = False

        self._build_form()
        self._build_actions()
        self._build_log()
        self.after(100, self._drain_log)

    def _build_form(self):
        frame = ttk.LabelFrame(self, text="Settings", padding=12)
        frame.pack(fill="x", padx=12, pady=(12, 6))
        frame.columnconfigure(1, weight=1)

        for row, (key, label, _kind) in enumerate(FIELDS):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 8))
            var = tk.StringVar(value=str(self.config_data.get(key, "")))
            self.vars[key] = var
            ttk.Entry(frame, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)

        hint = ttk.Label(
            frame,
            text="Tip: use {period} in the query template — it's replaced by the Period value.",
            foreground="#888",
            wraplength=560,
        )
        hint.grid(row=len(FIELDS), column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_actions(self):
        bar = ttk.Frame(self, padding=(12, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="Save Settings", command=self.save_config).pack(side="left")
        self.refresh_btn = ttk.Button(bar, text="Save & Refresh Now", command=self.refresh_now)
        self.refresh_btn.pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Copy Feed URL", command=self.copy_feed_url).pack(side="right")

    def _build_log(self):
        frame = ttk.LabelFrame(self, text="Output", padding=8)
        frame.pack(fill="both", expand=True, padx=12, pady=12)
        self.log = scrolledtext.ScrolledText(frame, height=12, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    # --- actions ---------------------------------------------------------
    def _collect_config(self) -> dict | None:
        new = dict(self.config_data)
        for key, label, kind in FIELDS:
            raw = self.vars[key].get().strip()
            if kind == "int":
                try:
                    new[key] = int(raw)
                except ValueError:
                    messagebox.showerror("Invalid value", f"'{label}' must be a whole number.")
                    return None
            else:
                new[key] = raw
        if "{period}" not in new["query_template"]:
            if not messagebox.askyesno(
                "No {period} placeholder",
                "The query template has no {period} placeholder, so the Period field "
                "won't be used. Save anyway?",
            ):
                return None
        return new

    def save_config(self) -> bool:
        new = self._collect_config()
        if new is None:
            return False
        with open(CONFIG_PATH, "w") as f:
            json.dump(new, f, indent=2)
            f.write("\n")
        self.config_data = new
        self._append_log(f"Saved settings to {CONFIG_PATH.name}.\n")
        return True

    def copy_feed_url(self):
        self.clipboard_clear()
        self.clipboard_append(FEED_URL)
        self._append_log(f"Copied feed URL to clipboard:\n{FEED_URL}\n")

    def refresh_now(self):
        if self.refresh_running:
            return
        if not self.save_config():
            return
        self.refresh_running = True
        self.refresh_btn.config(state="disabled", text="Refreshing…")
        self._append_log("\n--- Starting refresh ---\n")
        threading.Thread(target=self._run_refresh, daemon=True).start()

    def _run_refresh(self):
        try:
            proc = subprocess.Popen(
                [sys.executable, str(REPO_DIR / "generate_events.py")],
                cwd=REPO_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                self.log_queue.put(line)
            proc.wait()
            if proc.returncode == 0:
                self.log_queue.put("--- Refresh complete ✅ ---\n")
            else:
                self.log_queue.put(f"--- Refresh failed (exit {proc.returncode}) ❌ ---\n")
        except Exception as e:  # noqa: BLE001
            self.log_queue.put(f"Error: {e}\n")
        finally:
            self.log_queue.put("__DONE__")

    def _drain_log(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__DONE__":
                    self.refresh_running = False
                    self.refresh_btn.config(state="normal", text="Save & Refresh Now")
                else:
                    self._append_log(line)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _append_log(self, text: str):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    ControlPanel().mainloop()

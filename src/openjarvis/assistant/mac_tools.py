"""Tools Jarvis can use on your Mac. Each tool is a JSON schema plus a Python function.

Tools marked `confirm=True` ask you for approval in the HUD before they run.
"""

import base64
import ctypes
import json
import subprocess
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import memory


@dataclass
class Tool:
    name: str
    description: str
    params: dict  # property name -> JSON schema
    required: list[str]
    fn: Callable[..., "str | Image"]
    confirm: bool = False

    @property
    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": self.params,
            "required": self.required,
            "additionalProperties": False,
        }

    def execute_simple(self, args: object) -> tuple[str, bool]:
        """Run without the approval step, for tools that don't need one. Returns
        (text, is_error)."""
        output, is_error = execute(self.name, args, lambda *_: False)
        return (output.text if isinstance(output, Image) else output), is_error

    def validate(self, args: object) -> str | None:
        """Return an error message if args don't match the schema, else None."""
        if not isinstance(args, dict):
            return "arguments must be an object"
        for key in self.required:
            if key not in args:
                return f"missing required argument '{key}'"
        types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
        for key, value in args.items():
            if key not in self.params:
                return f"unknown argument '{key}'"
            expected = types.get(self.params[key].get("type"))
            if expected and not isinstance(value, expected):
                return f"argument '{key}' must be a {self.params[key]['type']}"
        return None


def _run(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _osascript(script: str) -> str:
    result = _run(["osascript", "-e", script])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


@dataclass
class Image:
    """A tool result the model looks at (a JPEG), with a line of text to go with it."""

    text: str
    jpeg: bytes

    @property
    def b64(self) -> str:
        return base64.b64encode(self.jpeg).decode()


# --- Tool implementations -----------------------------------------------------


def get_time(timezone: str = "") -> str:
    if not timezone:
        return datetime.now().astimezone().strftime("%A %d %B %Y, %H:%M %Z")
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Unknown time zone {timezone!r}. Use an IANA name such as 'Europe/Lisbon'."
    return f"{timezone}: " + datetime.now(zone).strftime("%A %d %B %Y, %H:%M %Z (UTC%z)")


_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
    48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def get_weather(location: str) -> str:
    geo = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search?count=1&name="
        + urllib.parse.quote(location)
    )
    if not geo.get("results"):
        return f"Couldn't find a place called {location}."
    place = geo["results"][0]
    query = urllib.parse.urlencode({
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": 1,
        "timezone": "auto",
    })
    data = _get_json(f"https://api.open-meteo.com/v1/forecast?{query}")
    cur, day = data["current"], data["daily"]
    return (
        f"{place['name']}, {place.get('country', '')}: "
        f"{_WEATHER_CODES.get(cur['weather_code'], 'unknown conditions')}, "
        f"{cur['temperature_2m']}°C (feels like {cur['apparent_temperature']}°C), "
        f"wind {cur['wind_speed_10m']} km/h. Today: high {day['temperature_2m_max'][0]}°C, "
        f"low {day['temperature_2m_min'][0]}°C, "
        f"{day['precipitation_probability_max'][0]}% chance of rain."
    )


def open_app(name: str) -> str:
    result = _run(["open", "-a", name])
    return f"Opened {name}." if result.returncode == 0 else f"Couldn't open {name}: {result.stderr.strip()}"


def open_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Only http(s) URLs are allowed."
    _run(["open", url])
    return f"Opened {url}."


def set_volume(level: int) -> str:
    level = max(0, min(100, level))
    _osascript(f"set volume output volume {level}")
    return f"Volume set to {level}%."


def media_control(action: str) -> str:
    verbs = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}
    if action not in verbs:
        return f"Unknown action {action}."
    for app in ("Spotify", "Music"):
        running = _osascript(f'application "{app}" is running')
        if running == "true":
            _osascript(f'tell application "{app}" to {verbs[action]}')
            return f"{app}: {action}."
    return "Neither Spotify nor Music is running."


def system_status() -> str:
    battery = _run(["pmset", "-g", "batt"]).stdout.strip().splitlines()
    uptime = _run(["uptime"]).stdout.strip()
    disk = _run(["df", "-h", "/"]).stdout.strip().splitlines()[-1].split()
    return (
        f"Battery: {battery[-1].strip() if battery else 'unknown'}\n"
        f"Uptime: {uptime}\n"
        f"Disk: {disk[3]} free of {disk[1]}"
    )


def web_search(query: str, news: bool = False) -> str:
    from ddgs import DDGS  # imported on first use; only the local brain needs it
    from ddgs.exceptions import DDGSException

    search = DDGS(timeout=10)
    # Searches fail now and then for no lasting reason, so try twice; news falls back to
    # ordinary web results.
    results, kind = [], "news" if news else "text"
    for attempt in (kind, "text"):
        try:
            results = getattr(search, attempt)(query, max_results=5)
        except DDGSException:
            continue
        if results:
            kind = attempt
            break
    if not results:
        return f"No results for {query!r}. Try different words."
    lines = []
    for i, r in enumerate(results, 1):
        when = f" ({r.get('source', '')}, {str(r.get('date', ''))[:10]})" if kind == "news" else ""
        lines.append(f"{i}. {r['title']}{when}\n   {r['body']}\n   {r.get('href') or r.get('url')}")
    # Snippets are often cut off before the answer, and small models rarely think to open a
    # page themselves, so include the text of the first readable result.
    for i, r in enumerate(results[:2], 1):
        try:
            text = _page_text(r.get("href") or r.get("url") or "", limit=2000)
        except Exception:  # noqa: BLE001 - the snippets are still useful on their own
            text = ""
        if text:
            lines.append(f"\nText of result {i}:\n{text}")
            break
    return "\n".join(lines)


def _page_text(url: str, limit: int) -> str:
    from ddgs import DDGS

    text = str(DDGS(timeout=10).extract(url, fmt="text_plain")["content"])
    # Menus, buttons and links are short lines; the content is in the long ones.
    lines = [line.strip() for line in text.splitlines()]
    content = "\n".join(line for line in lines if sum(c.isalnum() for c in line) >= 30)
    if len(content) > limit:  # the local model's context is small
        content = content[:limit] + "\n…(truncated)"
    return content


def read_webpage(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Only http(s) URLs are allowed."
    return _page_text(url, limit=4000) or "That page has no readable text (it may need JavaScript)."


# Longest side of screenshots, in pixels: enough to read text, small enough to stay quick
# for the local model.
SCREEN_SIZE = 1568


def _screen_access() -> bool:
    """Whether macOS lets Jarvis record the screen. Asks the user the first time."""
    cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    cg.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
    cg.CGRequestScreenCaptureAccess.restype = ctypes.c_bool
    return cg.CGPreflightScreenCaptureAccess() or cg.CGRequestScreenCaptureAccess()


def look_at_screen() -> Image | str:
    if not _screen_access():
        return (
            "No permission to see the screen. Tell the user to turn on Screen Recording for "
            "Jarvis (python3.12) in System Settings, Privacy and Security, then restart Jarvis."
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "screen.jpg"
        # -x: no shutter sound; -m: the main display only.
        result = _run(["screencapture", "-x", "-m", "-t", "jpg", str(path)])
        if result.returncode != 0 or not path.exists():
            raise RuntimeError(f"screencapture failed: {result.stderr.strip()}")
        _run(["sips", "-Z", str(SCREEN_SIZE), str(path)])
        return Image("This is what is on the user's screen right now.", path.read_bytes())


# Copies the selection in the frontmost app (Cmd+C) and puts the clipboard back as it was.
# JavaScript for Automation, because AppleScript can't save and restore every clipboard type.
_COPY_SELECTION = r"""
ObjC.import("AppKit");
const pb = $.NSPasteboard.generalPasteboard;
const saved = [];
const items = pb.pasteboardItems;
for (let i = 0; i < items.count; i++) {
  const item = items.objectAtIndex(i), types = item.types, copy = $.NSPasteboardItem.alloc.init;
  for (let j = 0; j < types.count; j++) {
    const data = item.dataForType(types.objectAtIndex(j));
    if (data) copy.setDataForType(data, types.objectAtIndex(j));
  }
  saved.push(copy);
}
const before = pb.changeCount;
Application("System Events").keystroke("c", {using: "command down"});
let text = "";
for (let i = 0; i < 20 && pb.changeCount === before; i++) delay(0.05);
if (pb.changeCount !== before) {
  text = ObjC.unwrap(pb.stringForType($.NSPasteboardTypeString)) || "";
  pb.clearContents;
  if (saved.length) pb.writeObjects($(saved));
}
text;
"""


def get_selected_text() -> str:
    result = _run(["osascript", "-l", "JavaScript", "-e", _COPY_SELECTION])
    if result.returncode != 0:
        if "1002" in result.stderr or "not allowed" in result.stderr:
            return (
                "No permission to copy the selection. Tell the user to turn on Accessibility for "
                "Jarvis (python3.12) in System Settings, Privacy and Security, then restart Jarvis."
            )
        raise RuntimeError(result.stderr.strip())
    text = result.stdout.strip()
    if not text:
        return "Nothing is selected in the front app. Try look_at_screen instead."
    if len(text) > 4000:
        text = text[:4000] + "\n…(truncated)"
    return f"Selected text:\n{text}"


def run_shell(command: str) -> str:
    result = _run(["/bin/zsh", "-lc", command], timeout=60)
    out = (result.stdout + result.stderr).strip() or "(no output)"
    if len(out) > 4000:
        out = out[:4000] + "\n…(truncated)"
    return f"exit code {result.returncode}\n{out}"


TOOLS: list[Tool] = [
    Tool(
        "get_time",
        "Get the current date and time, here or anywhere in the world. Use this (never the "
        "shell) for any question about the time, including in other countries or cities.",
        {
            "timezone": {
                "type": "string",
                "description": "IANA time zone for somewhere else, e.g. 'Europe/Lisbon' or "
                "'Asia/Tokyo'. Leave out for the user's own local time.",
            }
        },
        [],
        get_time,
    ),
    Tool(
        "get_weather",
        "Get current weather and today's forecast for a city.",
        {"location": {"type": "string", "description": "City name, e.g. 'London' or 'Austin'."}},
        ["location"],
        get_weather,
    ),
    Tool(
        "open_app",
        "Open (or bring to front) a macOS application by name.",
        {"name": {"type": "string", "description": "App name, e.g. 'Safari', 'Spotify', 'Notes'."}},
        ["name"],
        open_app,
    ),
    Tool(
        "open_url",
        "Open a web page in the default browser.",
        {"url": {"type": "string", "description": "Full http(s) URL."}},
        ["url"],
        open_url,
    ),
    Tool(
        "set_volume",
        "Set the Mac's output volume.",
        {"level": {"type": "integer", "description": "0 to 100."}},
        ["level"],
        set_volume,
    ),
    Tool(
        "media_control",
        "Control music playback in Spotify or Apple Music.",
        {"action": {"type": "string", "enum": ["play", "pause", "next", "previous"]}},
        ["action"],
        media_control,
    ),
    Tool(
        "get_selected_text",
        "Get the exact text the user has selected (highlighted) in the app in front. Use it "
        "first whenever the user mentions selected or highlighted text, or says 'this' about "
        "text: 'translate this', 'read this', 'explain what I've highlighted'. Much more "
        "accurate than reading a screenshot.",
        {},
        [],
        get_selected_text,
    ),
    Tool(
        "look_at_screen",
        "Take a screenshot of the user's main display and look at it. Use it whenever the "
        "user asks about what's on their screen or refers to something they're looking at "
        "('what's this error?', 'what do you think of this?'). You can see the screen with "
        "this tool, so never tell the user you can't; if you missed something, look again. "
        "For selected text use get_selected_text instead. Describe "
        "or answer from what you actually see in one or two sentences unless asked for more; "
        "don't read out everything, and say so if something is too small to make out.",
        {},
        [],
        look_at_screen,
    ),
    Tool("system_status", "Report battery, uptime and free disk space.", {}, [], system_status),
    Tool(
        "run_shell",
        "Run a zsh command on the user's Mac and return its output. The user must approve "
        "each command, so prefer the dedicated tools when one fits.",
        {"command": {"type": "string", "description": "The shell command to run."}},
        ["command"],
        run_shell,
        confirm=True,
    ),
    Tool(
        "remember",
        "Save a lasting fact about the user (preferences, people, projects, routines) to "
        "long-term memory. Use when the user shares something worth knowing in future sessions.",
        {"fact": {"type": "string", "description": "One concise fact, referring to the user as 'the user'."}},
        ["fact"],
        memory.remember,
    ),
    Tool(
        "forget",
        "Delete memory items containing the given text.",
        {"query": {"type": "string"}},
        ["query"],
        memory.forget,
    ),
]

# Given to the local brain only: Claude has its own server-side web search.
WEB_TOOLS: list[Tool] = [
    Tool(
        "web_search",
        "Search the web. Use it for anything current or that you're unsure of: news, scores, "
        "prices, releases, opening hours, facts that may have changed since your training. "
        "Never offer to search; just search. "
        "Set news to true for recent events and headlines. If the results don't contain the "
        "answer, search again with different words or read_webpage the best result before "
        "replying. Only state facts the results actually say; if they don't say, tell the "
        "user you couldn't find out. Answer in your own words and never read out URLs.",
        {
            "query": {"type": "string", "description": "What to search for."},
            "news": {"type": "boolean", "description": "Search recent news articles instead."},
        },
        ["query"],
        web_search,
    ),
    Tool(
        "read_webpage",
        "Read the text of a web page, such as a search result, when the snippets aren't "
        "enough to answer.",
        {"url": {"type": "string", "description": "Full http(s) URL."}},
        ["url"],
        read_webpage,
    ),
]

BY_NAME = {t.name: t for t in TOOLS + WEB_TOOLS}


def execute(
    name: str, args: object, confirm: Callable[[str, str], bool]
) -> tuple[str | Image, bool]:
    """Run a tool. Returns (result, is_error); the result is text or an Image."""
    tool = BY_NAME.get(name)
    if tool is None:
        return f"Unknown tool '{name}'.", True
    if error := tool.validate(args):
        return f"Invalid arguments: {error}", True
    if tool.confirm and not confirm(name, json.dumps(args)):
        return "The user declined to run this.", True
    try:
        return tool.fn(**cast(dict[str, Any], args)), False
    except Exception as e:  # noqa: BLE001 - report any failure to the model so it can recover
        return f"{type(e).__name__}: {e}", True

# J.A.R.V.I.S. voice assistant

The always-listening voice assistant, built on top of OpenJarvis. Say **"Hey Jarvis"**
(or hold <kbd>Space</kbd> in the HUD), and it answers out loud in a British voice, with an
arc-reactor HUD at http://127.0.0.1:8765.

```sh
uv sync --extra desktop --extra voice --extra tools-search --extra scheduler --extra assistant
uv run jarvis assistant
```

It thinks with **Claude** (set `ANTHROPIC_API_KEY` in `.env`) or the **local model**
OpenJarvis is set to use (`intelligence.default_model` in `~/.openjarvis/config.toml`;
override with `JARVIS_OLLAMA_MODEL`). It also works fully offline on the local model.

The Mac tools (time anywhere, open apps and sites, volume, music, system status, selected
text) are also OpenJarvis tools, so `jarvis chat --voice` and `jarvis ask` can use them
with the `orchestrator` agent.

The code is in `src/openjarvis/assistant/`.

## What Jarvis can do

Time, weather (Open-Meteo, no key), open apps and websites, set volume, control
Spotify/Music, system status, look at your screen and talk about it (screenshots stay on
your Mac with the local brain), web search and reading web pages (both brains; the local
brain uses DuckDuckGo, no key needed), run shell commands
(**every command needs your approval in the HUD**), and remember facts about you across
sessions (stored in `~/.jarvis/memory.json`).

With no internet, Jarvis keeps working on the local brain: if Claude can't be reached it
switches over by itself. Only weather and web search need a connection.

Add your own tools in `src/openjarvis/assistant/mac_tools.py`: write a function, add a `Tool(...)` entry, done.
Both brains pick it up automatically.

## Commands

| Action | Command |
|---|---|
| Stop | `launchctl kill TERM gui/$(id -u)/com.phillip.jarvis` |
| Start | `launchctl kickstart gui/$(id -u)/com.phillip.jarvis` |
| Restart (after editing `.env` or the code) | `launchctl kickstart -k gui/$(id -u)/com.phillip.jarvis` |
| Turn off (won't start at login) | `launchctl bootout gui/$(id -u)/com.phillip.jarvis` |
| Turn on (starts now and at login) | `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.phillip.jarvis.plist` |
| Is it running? | `launchctl print gui/$(id -u)/com.phillip.jarvis \| grep "state ="` |
| Watch the log | `tail -f ~/.jarvis/jarvis.log` |
| Open the HUD | `open http://127.0.0.1:8765` |

A stopped Jarvis starts again at your next login. Use **Turn off** to keep it off.

## Running it by hand

Useful for watching output while testing. Only one copy can run at a time, so turn
off the background one first:

```sh
launchctl bootout gui/$(id -u)/com.phillip.jarvis
cd ~/jarvis && uv run jarvis assistant   # Ctrl+C to stop
```

When you're done, use **Turn on** to bring the background copy back.

## Shortcuts (optional)

Add these to `~/.zshrc`, then open a new Terminal window:

```sh
alias jarvis-start='launchctl kickstart gui/$(id -u)/com.phillip.jarvis'
alias jarvis-stop='launchctl kill TERM gui/$(id -u)/com.phillip.jarvis'
alias jarvis-restart='launchctl kickstart -k gui/$(id -u)/com.phillip.jarvis'
alias jarvis-log='tail -f ~/.jarvis/jarvis.log'
```

## Troubleshooting

- **Jarvis doesn't hear you:** check System Settings → Privacy & Security →
  Microphone and make sure Python is turned on, then restart Jarvis.
- **Jarvis can't see your screen:** the first time you ask, macOS asks for Screen
  Recording permission. If you missed it, turn on Python in System Settings → Privacy &
  Security → Screen & System Audio Recording, then restart Jarvis.
- **Wake word stops working overnight:** the Mac went to sleep. Turn on "Prevent
  automatic sleeping when the display is off" in System Settings → Energy.
- **"Address already in use" when running by hand:** the background copy is still
  running. Turn it off first (see above).
- **"Translate what I've selected" doesn't work:** Jarvis copies your selection, which
  needs Accessibility permission. Turn on Python in System Settings → Privacy & Security
  → Accessibility, then restart Jarvis.
- **Something else is wrong:** check `~/.jarvis/jarvis.log`.

## Tuning

All settings are in `.env` (see `.env.example`). Common tweaks:

- **Faster / smarter Claude:** `JARVIS_CLAUDE_EFFORT=low|medium|high`. Change the model with `JARVIS_CLAUDE_MODEL`.
- **Wake word too sensitive / not sensitive enough:** `JARVIS_WAKE_THRESHOLD` (0–1).
- **Jarvis cuts you off / waits too long after you finish:** `JARVIS_END_SILENCE`
  (seconds of pause, default 1.5). Speech is detected with Silero VAD, so background
  noise and quiet words don't confuse it.
- **Speed vs. memory:** Jarvis starts loading the local model as soon as it hears "Hey
  Jarvis", and keeps it loaded for `JARVIS_OLLAMA_KEEP_ALIVE` after you last spoke
  (default `30m`; `-1` keeps it loaded always). Loaded, answers start in about 2 s; after
  it unloads, the first answer takes about 8 s. It uses about 10 GB of memory while loaded.
- **Local brain making things up from web results:** set `JARVIS_OLLAMA_THINK=true`.
  It reasons before answering from search results: far more reliable, but those answers
  take 15-45 s instead of about 8.
- **Voice:** `JARVIS_VOICE` picks the Kokoro voice. British men: `bm_george` (default),
  `bm_lewis`, `bm_daniel`, `bm_fable`; British women: `bf_emma`, `bf_alice`, `bf_isabella`,
  `bf_lily`. `JARVIS_VOICE_SPEED` adjusts pace. Set `JARVIS_TTS=say` to use the built-in
  macOS voices instead.
- **Memory:** speech models load only while you're talking to Jarvis and unload after
  `JARVIS_KEEP_LOADED` seconds (default 60), so idle Jarvis uses ~200 MB.

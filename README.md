# J.A.R.V.I.S.

A personal voice assistant for the Mac. Say **"Hey Jarvis"** and talk: it answers out loud
in a British voice, controls your Mac, reads your screen and remembers what you tell it,
with an arc-reactor HUD in your browser. It runs on a local model, so it works offline.

Built on top of [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) from Stanford.

## Run and stop

Jarvis runs in the background and starts by itself at login.

| Action | Command |
|---|---|
| **Start** (now and at every login) | `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.phillip.jarvis.plist` |
| **Stop** (stays off until your next login) | `launchctl bootout gui/$(id -u)/com.phillip.jarvis` |
| Restart (after changing settings or code) | `launchctl kickstart -k gui/$(id -u)/com.phillip.jarvis` |
| Is it running? | `launchctl print gui/$(id -u)/com.phillip.jarvis \| grep "state ="` |
| Open the HUD | `open http://127.0.0.1:8765` |
| Watch the log | `tail -f ~/.jarvis/jarvis.log` |

Use **Stop** rather than killing the process: macOS restarts a killed Jarvis straight away.

To run it in a terminal instead (stop the background copy first; only one can run at a time):

```sh
cd ~/jarvis && uv run jarvis assistant      # Ctrl+C to stop
```

## Talking to it

| Do this | To |
|---|---|
| Say **"Hey Jarvis"**, then speak | Ask hands-free; it waits until you've finished |
| Hold **Space** in the HUD | Push-to-talk (also interrupts Jarvis) |
| Type in the HUD | Ask without speaking |
| **Claude / Local** buttons | Switch brains |
| **Reset** | Start a fresh conversation (memories are kept) |

## What it can do

- Tell the time here or anywhere in the world, and the weather anywhere
- Open apps and websites, set the volume, control Spotify or Apple Music
- Report battery, uptime and free disk space
- Look at your screen and talk about it; read, translate or explain text you've selected
- Search the web and read pages
- Run shell commands (each one needs your approval in the HUD)
- Remember facts about you between conversations

It thinks with a local model through Ollama (`mimo-v2.6:9b`), or with Claude if you add
`ANTHROPIC_API_KEY` to `.env`. Without internet it keeps working on the local model; only
weather and web search need a connection.

## Also included: OpenJarvis

Everything from OpenJarvis still works, and can use the Mac tools above:

```sh
uv run jarvis chat --voice             # spoken chat in the terminal
uv run jarvis ask "What time is it in Tokyo?"
uv run jarvis gui                      # OpenJarvis's browser interface
uv run jarvis doctor                   # check the setup
```

See [OPENJARVIS_README.md](OPENJARVIS_README.md) and the
[OpenJarvis docs](https://open-jarvis.github.io/OpenJarvis/) for its other features.

## Setup

```sh
brew install uv portaudio ollama
brew services start ollama && ollama pull mimo-v2.6:9b
uv sync --extra desktop --extra voice --extra tools-search --extra scheduler --extra assistant
uv run jarvis init --engine ollama     # then set default_model in ~/.openjarvis/config.toml
uv run jarvis assistant
```

macOS asks for permission the first time Jarvis uses the microphone, the screen
(Screen Recording) and your selected text (Accessibility). Allow each one for Python.

Settings, tuning and troubleshooting: [ASSISTANT.md](ASSISTANT.md). The voice assistant's
code is in `src/openjarvis/assistant/`.

## Credits

This project is built on **[OpenJarvis](https://github.com/open-jarvis/OpenJarvis)**,
"Personal AI, On Personal Devices", by Jon Saad-Falcon, Avanika Narayan, Robby Manihani,
Tanvir Bhathal, Herumb Shandilya, Hakki Orhun Akengin, Gabriel Bo, Andrew Park,
Matthew Hart, Caia Costello, Chuan Li, Christopher Ré, Azalia Mirhoseini and the OpenJarvis
contributors. It is developed at [Hazy Research](https://hazyresearch.stanford.edu/) and
the [Scaling Intelligence Lab](https://scalingintelligence.stanford.edu/) at
[Stanford SAIL](https://ai.stanford.edu/), as part of
[Intelligence Per Watt](https://www.intelligence-per-watt.ai/).
Read the [paper](https://arxiv.org/abs/2605.17172) or visit the
[project site](https://openjarvis.stanford.edu/).

```bibtex
@misc{saadfalcon2026openjarvispersonalaipersonal,
      title={OpenJarvis: Personal AI, On Personal Devices},
      author={Jon Saad-Falcon and Avanika Narayan and Robby Manihani and Tanvir Bhathal and Herumb Shandilya and Hakki Orhun Akengin and Gabriel Bo and Andrew Park and Matthew Hart and Caia Costello and Chuan Li and Christopher Ré and Azalia Mirhoseini},
      year={2026},
      eprint={2605.17172},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.17172},
}
```

The voice assistant also stands on these open-source projects:
[Ollama](https://ollama.com/) (local models),
[MiMo](https://huggingface.co/XiaomiMiMo) by Xiaomi (the local model),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (speech recognition),
[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) and
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) (voice),
[openWakeWord](https://github.com/dscripka/openWakeWord) ("Hey Jarvis"),
[Silero VAD](https://github.com/snakers4/silero-vad) (knowing when you've finished talking),
[DDGS](https://github.com/deedy5/ddgs) (web search) and
[Open-Meteo](https://open-meteo.com/) (weather).

## License

[Apache 2.0](LICENSE), the same as OpenJarvis. Copyright 2025 The OpenJarvis Authors;
the voice assistant additions are released under the same license.

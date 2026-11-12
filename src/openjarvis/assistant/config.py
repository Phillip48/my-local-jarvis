"""Settings, read from environment variables (and a .env file in the project root)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Quiet third-party startup chatter (Whisper's backend tips, Hugging Face download notices).
os.environ.setdefault("CT2_VERBOSE", "-1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# Which brain to start with: "claude" or "ollama". Switchable at runtime from the HUD.
BRAIN = os.getenv("JARVIS_BRAIN", "claude")
CLAUDE_MODEL = os.getenv("JARVIS_CLAUDE_MODEL", "claude-opus-5-5")
# Voice is latency-sensitive, so default to low effort. Raise to "medium"/"high" for harder tasks.
CLAUDE_EFFORT = os.getenv("JARVIS_CLAUDE_EFFORT", "low")
if CLAUDE_EFFORT not in ("low", "medium", "high", "xhigh", "max"):
    raise SystemExit(f"JARVIS_CLAUDE_EFFORT must be low, medium, high, xhigh or max (got {CLAUDE_EFFORT!r})")
if BRAIN not in ("claude", "ollama"):
    raise SystemExit("JARVIS_BRAIN must be claude or ollama")


def _openjarvis_model() -> str:
    """The model OpenJarvis is set to use (~/.openjarvis/config.toml), if it's an Ollama one."""
    try:
        from openjarvis.core.config import load_config

        cfg = load_config()
        if cfg.engine.default == "ollama" and cfg.intelligence.default_model:
            return cfg.intelligence.default_model
    except Exception:  # noqa: BLE001 - fall back to the default below
        pass
    return "mimo-v2.6:9b"


OLLAMA_MODEL = os.getenv("JARVIS_OLLAMA_MODEL") or _openjarvis_model()
# Tokens of conversation the local model can see; Ollama's default (4096) fills up after
# a web search or two.
OLLAMA_CONTEXT = int(os.getenv("JARVIS_OLLAMA_CONTEXT", "8192"))
# Let the local model reason before answering from tool results (e.g. web searches). Much
# less likely to make things up, but those answers take 15-45 s instead of about 8.
OLLAMA_THINK = _flag("JARVIS_OLLAMA_THINK", False)
# How long the local model stays in memory after you last spoke to Jarvis (Ollama
# duration, e.g. "30m", "2h", or "-1" for always). Loading it takes about 20 s, but it
# holds about 10 GB of memory while loaded.
OLLAMA_KEEP_ALIVE = os.getenv("JARVIS_OLLAMA_KEEP_ALIVE", "30m")

# Speech
WHISPER_MODEL = os.getenv("JARVIS_WHISPER_MODEL", "small.en")
# "kokoro": natural neural voice (local). "say": built-in macOS voices (robotic, no download).
TTS = os.getenv("JARVIS_TTS", "kokoro")
# Kokoro British voices: bm_george, bm_lewis, bm_daniel, bm_fable (bf_* are female).
# For TTS=say, any voice from `say -v '?'`, e.g. Daniel.
VOICE = os.getenv("JARVIS_VOICE", "bm_george" if TTS == "kokoro" else "Daniel")
VOICE_LANG = "en-gb" if VOICE.startswith(("bm_", "bf_")) else "en-us"
VOICE_SPEED = float(os.getenv("JARVIS_VOICE_SPEED", "1.0"))  # Kokoro
VOICE_RATE = int(os.getenv("JARVIS_VOICE_RATE", "190"))  # say, words per minute
WAKE_WORD = _flag("JARVIS_WAKE_WORD", True)
WAKE_THRESHOLD = float(os.getenv("JARVIS_WAKE_THRESHOLD", "0.5"))
# Seconds of silence that mean you have finished talking (after the wake word).
END_SILENCE = float(os.getenv("JARVIS_END_SILENCE", "1.5"))

# Your name, so Jarvis can address you properly.
USER_NAME = os.getenv("JARVIS_USER_NAME", "sir")

HOST = os.getenv("JARVIS_HOST", "127.0.0.1")
PORT = int(os.getenv("JARVIS_PORT", "8765"))

DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", Path.home() / ".jarvis"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "memory.json"
MODELS_DIR = Path(os.getenv("JARVIS_MODELS_DIR", Path.home() / ".jarvis" / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Seconds the speech models stay in memory after use. They reload in well under a second,
# and loading starts as soon as you begin talking, so you won't notice.
KEEP_LOADED = float(os.getenv("JARVIS_KEEP_LOADED", "60"))

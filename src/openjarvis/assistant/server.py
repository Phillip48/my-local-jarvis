"""Ties everything together: mic -> speech-to-text -> brain -> speech, with a web HUD."""

import asyncio
import itertools
import json
import os
import subprocess
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from . import config, memory
from .audio import Mic, Speaker, Transcriber
from .brains import BRAINS, Events

HUD = Path(__file__).parent / "hud" / "index.html"


NO_KEY = "Claude needs an API key: add ANTHROPIC_API_KEY to your .env file and restart me."


class Interrupted(Exception):
    pass


def claude_configured() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


class Jarvis:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[WebSocket] = set()
        self.state = "booting"
        self.brains: dict = {}
        self.notice = ""  # shown in the HUD once it connects
        brain = config.BRAIN
        if brain == "claude" and not claude_configured():
            brain = "ollama"
            self.notice = NO_KEY + " Using the local brain for now."
            print(self.notice, flush=True)
        self.brain = self._get_brain(brain)
        self.turn = 0  # bumped on every interruption; stale turns notice and stop
        self.turn_lock = threading.Lock()
        self.confirm_ids = itertools.count(1)
        self.pending_confirms: dict[int, list] = {}  # id -> [Event, approved]

        print("Checking speech recognition model (the first run downloads it)…", flush=True)
        self.stt = Transcriber()
        self.speaker = Speaker(on_start=self._on_speaking, on_idle=self._on_speech_done)
        self.mic = Mic(on_wake=self._on_wake, on_utterance=self._on_utterance, on_level=self._on_level)

    # --- HUD plumbing ---------------------------------------------------------

    def broadcast(self, msg: dict) -> None:
        if self.loop is None:
            return
        data = json.dumps(msg)

        async def send_all() -> None:
            for ws in list(self.clients):
                try:
                    await ws.send_text(data)
                except Exception:  # noqa: BLE001 - a closed tab shouldn't break the others
                    self.clients.discard(ws)

        asyncio.run_coroutine_threadsafe(send_all(), self.loop)

    def set_state(self, state: str) -> None:
        self.state = state
        self.broadcast({"type": "state", "state": state})

    def status(self) -> dict:
        return {
            "type": "status",
            "state": self.state,
            "brain": self.brain.name,
            "model": self.brain.label,
            "wake": self.mic.wake_enabled,
            "mic": self.mic.available,
            "memories": len(memory.load()),
            "user": config.USER_NAME,
        }

    # --- Audio callbacks ------------------------------------------------------

    def _on_level(self, level: float) -> None:
        self.broadcast({"type": "level", "level": round(level, 3)})

    def _on_speaking(self) -> None:
        self.mic.muted = True
        self.set_state("speaking")

    def _on_speech_done(self) -> None:
        self.mic.muted = False

    def _on_wake(self) -> None:
        if self.state != "idle":
            return
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
        self.set_state("listening")
        self.stt.warm()
        self.brain.warm()
        self.mic.start_recording(auto_stop=True)

    def _on_utterance(self, audio: np.ndarray) -> None:
        self.set_state("thinking")
        text = self.stt(audio)
        if not text:
            self.set_state("idle")
            return
        self.handle_text(text)

    # --- Conversation ---------------------------------------------------------

    def interrupt(self) -> None:
        self.turn += 1
        self.speaker.stop()
        for pending in self.pending_confirms.values():
            pending[0].set()  # unblock any waiting approval as "declined"

    def handle_text(self, text: str) -> None:
        turn = self.turn
        with self.turn_lock:
            if turn != self.turn:
                return
            self.broadcast({"type": "message", "role": "user", "text": text})
            self.set_state("thinking")
            self.speaker.warm()
            reply: list[str] = []

            def on_text(delta: str) -> None:
                if turn != self.turn:
                    raise Interrupted
                reply.append(delta)
                self.broadcast({"type": "delta", "text": delta})
                self.speaker.feed(delta)

            def on_tool(name: str, args: dict) -> None:
                if turn != self.turn:
                    raise Interrupted
                self.broadcast({"type": "tool", "name": name, "args": args})

            try:
                try:
                    self.brain.respond(text, Events(on_text, on_tool, self.confirm))
                except anthropic.APIConnectionError:
                    if reply or not self._go_local():
                        raise
                    self.brain.respond(text, Events(on_text, on_tool, self.confirm))
                self.speaker.flush()
                self.broadcast({"type": "done"})
                self.speaker.wait()
            except Interrupted:
                self.broadcast({"type": "done"})
                return
            except anthropic.AuthenticationError:
                self._fail("Claude rejected the API key. Check ANTHROPIC_API_KEY in your .env file.")
            except anthropic.APIConnectionError:
                self._fail("I can't reach Claude. Check the internet connection, or switch me to the local brain.")
            except anthropic.APIStatusError as e:
                self._fail(f"Claude returned an error: {e.message}")
            except Exception as e:  # noqa: BLE001 - speak any other failure instead of dying
                if "connect" in str(e).lower() and self.brain.name == "ollama":
                    self._fail("I can't reach Ollama. Start it with: brew services start ollama.")
                else:
                    self._fail(f"Something went wrong: {e}")
            if turn == self.turn:
                self.set_state("idle")
                self.broadcast({"type": "memories", "count": len(memory.load())})

    def _go_local(self) -> bool:
        """Claude is unreachable (no internet): switch to the local brain. False if Ollama
        isn't running either."""
        brain = self._get_brain("ollama")
        try:
            brain.client.list()
        except Exception:  # noqa: BLE001 - Ollama not running
            return False
        self.brain = brain
        print("Can't reach Claude: switched to the local brain.", flush=True)
        self.broadcast({"type": "error", "text": "No internet, so I've switched to the local brain."})
        self.broadcast(self.status())
        return True

    def _fail(self, message: str) -> None:
        print(f"[error] {message}", flush=True)
        self.broadcast({"type": "error", "text": message})
        self.speaker.say("I'm afraid something went wrong, " + config.USER_NAME + ".")
        self.speaker.wait()

    def confirm(self, tool: str, args: str) -> bool:
        if not self.clients:
            return False
        cid = next(self.confirm_ids)
        pending = self.pending_confirms[cid] = [threading.Event(), False]
        self.broadcast({"type": "confirm", "id": cid, "tool": tool, "args": json.loads(args)})
        self.speaker.say("I'll need your approval for that.")
        pending[0].wait(timeout=120)
        del self.pending_confirms[cid]
        self.broadcast({"type": "confirm_closed", "id": cid})
        return pending[1]

    def _get_brain(self, name: str):
        if name not in self.brains:
            self.brains[name] = BRAINS[name]()
        return self.brains[name]

    # --- Commands from the HUD ------------------------------------------------

    def command(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind in ("ptt_start", "ptt_stop") and not self.mic.available:
            return
        if kind == "ptt_start":
            self.interrupt()
            self.set_state("listening")
            self.stt.warm()
            self.brain.warm()
            self.mic.start_recording(auto_stop=False)
        elif kind == "ptt_stop":
            self.mic.stop_recording()
        elif kind == "text" and msg.get("text", "").strip():
            self.interrupt()
            threading.Thread(target=self.handle_text, args=(msg["text"].strip(),), daemon=True).start()
        elif kind == "confirm" and msg.get("id") in self.pending_confirms:
            pending = self.pending_confirms[msg["id"]]
            pending[1] = bool(msg.get("approved"))
            pending[0].set()
        elif kind == "brain" and msg.get("name") in BRAINS:
            if msg["name"] == "claude" and not claude_configured():
                self.broadcast({"type": "error", "text": NO_KEY})
                return
            self.interrupt()
            self.brain = self._get_brain(msg["name"])
            self.set_state("idle")
            self.broadcast(self.status())
        elif kind == "wake":
            self.mic.set_wake(bool(msg.get("enabled")))
            self.broadcast(self.status())
        elif kind == "reset":
            self.interrupt()
            self.brain.reset()
            self.set_state("idle")
            self.broadcast({"type": "reset"})

    def _on_mic_ready(self) -> None:
        print("Microphone connected.", flush=True)
        self.broadcast(self.status())

    def boot(self) -> None:
        self.mic.start(on_ready=self._on_mic_ready)
        if not self.mic.available:
            print("No microphone found: typing works now, voice will start when you plug one in.", flush=True)
        self.set_state("idle")
        greeting = f"Good day, {config.USER_NAME}. All systems are online."
        self.broadcast({"type": "message", "role": "assistant", "text": greeting})
        self.speaker.say(greeting)


def create_app(jarvis: Jarvis) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        jarvis.loop = asyncio.get_running_loop()
        jarvis.boot()
        url = f"http://{config.HOST}:{config.PORT}"
        print(f"J.A.R.V.I.S. online at {url}", flush=True)
        webbrowser.open(url)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(HUD)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        jarvis.clients.add(socket)
        await socket.send_text(json.dumps(jarvis.status()))
        if jarvis.notice:
            await socket.send_text(json.dumps({"type": "error", "text": jarvis.notice}))
        try:
            while True:
                try:
                    msg = json.loads(await socket.receive_text())
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict):
                    await asyncio.to_thread(jarvis.command, msg)
        except WebSocketDisconnect:
            jarvis.clients.discard(socket)

    return app


def run() -> None:
    jarvis = Jarvis()
    uvicorn.run(create_app(jarvis), host=config.HOST, port=config.PORT, log_level="warning")

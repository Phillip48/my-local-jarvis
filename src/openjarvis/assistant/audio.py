"""Microphone input (wake word + recording), speech-to-text, and text-to-speech."""

import queue
import re
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import numpy as np
import sounddevice as sd

from . import config

T = TypeVar("T")

RATE = 16000
BLOCK = 1280  # 80 ms, the frame size openwakeword expects
SPEECH_ON, SPEECH_HOLD = 0.5, 0.3  # VAD probability to start / keep counting as speech
WAIT_FOR_SPEECH = 6.0  # give up if nothing is said this long after the wake word
MAX_RECORDING = 30.0


class Mic:
    """One always-open input stream, shared by the wake-word detector and the recorder.

    `on_utterance(audio)` fires with float32 audio when a recording finishes, either
    because push-to-talk was released or (after a wake word) the user stopped talking.
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        on_utterance: Callable[[np.ndarray], None],
        on_level: Callable[[float], None],
    ) -> None:
        self.on_wake, self.on_utterance, self.on_level = on_wake, on_utterance, on_level
        self.wake_enabled = config.WAKE_WORD
        self.muted = False  # set while Jarvis is talking, so he doesn't wake himself up
        self._recording = False
        self._auto_stop = False
        self._buf: list[np.ndarray] = []
        self._blocks: queue.Queue[np.ndarray] = queue.Queue()
        self._lock = threading.Lock()
        self._wake_model = None
        self._stream: sd.InputStream | None = None
        from openwakeword import utils
        from openwakeword.vad import VAD

        utils.download_models(["hey_jarvis"])  # also fetches the Silero VAD model
        # A neural speech detector decides when you've finished, so quiet words and
        # background noise don't end the recording early the way a volume check would.
        self._vad = VAD()
        if self.wake_enabled:
            self._load_wake_model()

    def _load_wake_model(self) -> None:
        from openwakeword.model import Model

        self._wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    def set_wake(self, enabled: bool) -> None:
        if enabled and self._wake_model is None:
            self._load_wake_model()
        self.wake_enabled = enabled

    available = False

    def start(self, on_ready: Callable[[], None]) -> None:
        """Open the microphone, retrying in the background until one is plugged in."""
        threading.Thread(target=self._process, daemon=True).start()

        if self._open():
            return

        def retry() -> None:
            while True:
                time.sleep(3)
                # PortAudio only sees devices that existed when it started, so rescan.
                sd._terminate()
                sd._initialize()
                if self._open():
                    on_ready()
                    return

        threading.Thread(target=retry, daemon=True).start()

    def _open(self) -> bool:
        try:
            stream = sd.InputStream(
                samplerate=RATE, channels=1, dtype="int16", blocksize=BLOCK,
                callback=lambda data, *_: self._blocks.put(data[:, 0].copy()),
            )
            stream.start()
        except sd.PortAudioError:
            return False
        self._stream = stream
        self.available = True
        return True

    def start_recording(self, auto_stop: bool) -> None:
        with self._lock:
            self._buf = []
            self._auto_stop = auto_stop
            self._recording = True
            self._heard_speech = False
            self._in_speech = False
            self._last_speech = 0.0  # seconds into the recording, measured in audio
            self._vad.reset_states()

    def stop_recording(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.int16)
        self.on_level(0.0)
        threading.Thread(
            target=self.on_utterance, args=(audio.astype(np.float32) / 32768.0,), daemon=True
        ).start()

    def _process(self) -> None:
        while True:
            block = self._blocks.get()
            if self._recording:
                self._record_block(block)
            elif self.wake_enabled and not self.muted and self._wake_model is not None:
                scores: dict = self._wake_model.predict(block)  # type: ignore[assignment]
                score = scores["hey_jarvis"]
                if score >= config.WAKE_THRESHOLD:
                    self._wake_model.reset()
                    self.on_wake()

    def _record_block(self, block: np.ndarray) -> None:
        with self._lock:
            self._buf.append(block)
            now = len(self._buf) * BLOCK / RATE  # audio time, so a busy CPU can't skew it
        rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
        self.on_level(min(1.0, rms / 3000))
        if not self._auto_stop:
            return
        prob = float(self._vad.predict(block, frame_size=BLOCK // 2))
        self._in_speech = prob > (SPEECH_HOLD if self._in_speech else SPEECH_ON)
        if self._in_speech:
            self._heard_speech = True
            self._last_speech = now
        silent_for = now - self._last_speech
        if (
            (self._heard_speech and silent_for > config.END_SILENCE)  # finished talking
            or (not self._heard_speech and now > WAIT_FOR_SPEECH)  # never started
            or now > MAX_RECORDING
        ):
            self.stop_recording()


class LazyModel(Generic[T]):
    """Holds a model in memory only while it's being used.

    `warm()` starts loading in the background, `get()` returns the model (loading it if
    needed), and `release()` unloads it after `config.KEEP_LOADED` seconds of disuse.
    """

    def __init__(self, loader: Callable[[], T]) -> None:
        self._loader = loader
        self._model: T | None = None
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def warm(self) -> None:
        self._cancel_unload()
        threading.Thread(target=self.get, daemon=True).start()

    def get(self) -> T:
        self._cancel_unload()
        with self._lock:
            if self._model is None:
                self._model = self._loader()
            return self._model

    def release(self) -> None:
        self._cancel_unload()
        self._timer = threading.Timer(config.KEEP_LOADED, self._unload)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_unload(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _unload(self) -> None:
        with self._lock:
            self._model = None


class Transcriber:
    """Whisper speech-to-text. Call `warm()` when recording starts so it's ready in time."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel, download_model

        try:  # use the cached copy without going online, so Jarvis starts offline
            path = download_model(config.WHISPER_MODEL, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet: fetch it once
            path = download_model(config.WHISPER_MODEL)
        self._model = LazyModel(lambda: WhisperModel(path, device="cpu", compute_type="int8"))
        self.warm = self._model.warm

    def __call__(self, audio: np.ndarray) -> str:
        try:
            if len(audio) < RATE * 0.3:
                return ""
            model = self._model.get()
            segments, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            self._model.release()


# --- Voices ---------------------------------------------------------------------------------
# A voice turns a sentence into a playable clip (`render`), plays it (`play`, blocking), and
# can be cut off mid-clip (`stop`). Rendering runs one sentence ahead of playback.

KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")


class KokoroVoice:
    """Natural-sounding neural voice (Kokoro, ~90 MB), running locally."""

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro

        for name in KOKORO_FILES:
            path = config.MODELS_DIR / name
            if not path.exists():
                print(f"Downloading voice model {name}…", flush=True)
                tmp = path.with_suffix(".part")
                urllib.request.urlretrieve(KOKORO_URL + name, tmp)
                tmp.rename(path)
        model_path, voices_path = (str(config.MODELS_DIR / n) for n in KOKORO_FILES)

        def load() -> Kokoro:
            kokoro = Kokoro(model_path, voices_path)
            # The first synthesis after loading is ~1s slower; pay that cost now, in the
            # background, instead of on the first sentence of a reply.
            kokoro.create("Ready.", voice=config.VOICE, lang=config.VOICE_LANG)
            return kokoro

        self._model = LazyModel(load)
        available = self._model.get().get_voices()
        self._model.release()
        if config.VOICE not in available:
            raise ValueError(f"unknown Kokoro voice {config.VOICE!r}; choose from {', '.join(available)}")
        self.warm, self.release = self._model.warm, self._model.release

    def render(self, text: str) -> tuple[np.ndarray, int]:
        return self._model.get().create(
            text, voice=config.VOICE, speed=config.VOICE_SPEED, lang=config.VOICE_LANG
        )

    def play(self, clip: tuple[np.ndarray, int]) -> None:
        sd.play(*clip)
        sd.wait()

    def stop(self) -> None:
        sd.stop()


class SayVoice:
    """The built-in macOS `say` voices: robotic, but needs nothing extra."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None

    def warm(self) -> None: ...
    def release(self) -> None: ...

    def render(self, text: str) -> str:
        return text

    def play(self, clip: str) -> None:
        self._proc = subprocess.Popen(["say", "-v", config.VOICE, "-r", str(config.VOICE_RATE), clip])
        self._proc.wait()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def load_voice() -> KokoroVoice | SayVoice:
    if config.TTS == "kokoro":
        try:
            return KokoroVoice()
        except Exception as e:  # noqa: BLE001 - any failure: fall back rather than go mute
            print(f"Kokoro voice unavailable ({e}); falling back to macOS `say`.", flush=True)
            config.VOICE = "Daniel"
    return SayVoice()


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
OPENING_WORDS = 5  # start talking at the first comma once there are this many words
_MARKDOWN = re.compile(r"[*_#`>|]+|\[(.*?)\]\(.*?\)")
_END = object()  # marks the end of a reply in the queues

# Units the voice would otherwise read literally ("degrees C", "km slash h").
_SPOKEN = [
    (re.compile(r"\s*°\s*C\b"), " degrees"),
    (re.compile(r"\s*°\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"\s*°"), " degrees"),
    (re.compile(r"(?<=\d)\s*km/h\b"), " kilometres per hour"),
    (re.compile(r"(?<=\d)\s*mph\b"), " miles per hour"),
    (re.compile(r"(?<=\d)\s*(?:Ti|TB)\b"), " terabytes"),
    (re.compile(r"(?<=\d)\s*(?:Gi|GB)\b"), " gigabytes"),
    (re.compile(r"(?<=\d)\s*(?:Mi|MB)\b"), " megabytes"),
]


def speakable(text: str) -> str:
    text = _MARKDOWN.sub(r"\1", text)
    for pattern, spoken in _SPOKEN:
        text = pattern.sub(spoken, text)
    return text.strip()


class Speaker:
    """Speaks a reply one sentence at a time, as it streams in from the brain."""

    def __init__(self, on_start: Callable[[], None], on_idle: Callable[[], None]) -> None:
        self.on_start, self.on_idle = on_start, on_idle
        self.voice = load_voice()
        self._pending = ""
        self._started_reply = False  # has any of the current reply been queued yet?
        self._generation = 0  # bumped by stop(); anything tagged with an older one is dropped
        self._text: queue.Queue[tuple[int, Any]] = queue.Queue()
        self._clips: queue.Queue[tuple[int, Any]] = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._render_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()

    def warm(self) -> None:
        """Get the voice ready while the brain is still thinking."""
        self.voice.warm()

    def feed(self, text: str) -> None:
        """Add streamed text; complete sentences are spoken immediately."""
        self._pending += text
        parts = _SENTENCE_END.split(self._pending)
        for sentence in parts[:-1]:
            self._say(sentence)
        self._pending = parts[-1]
        if not self._started_reply:
            # Don't wait for a long first sentence: begin at its first natural pause.
            for match in _CLAUSE_END.finditer(self._pending):
                if len(self._pending[: match.start()].split()) >= OPENING_WORDS:
                    self._say(self._pending[: match.start()])
                    self._pending = self._pending[match.end() :]
                    break

    def flush(self) -> None:
        self._say(self._pending)
        self._pending = ""

    def say(self, text: str) -> None:
        self.feed(text)
        self.flush()

    def _say(self, sentence: str) -> None:
        sentence = speakable(sentence)
        if sentence:
            self._started_reply = True
            self._idle.clear()
            self._text.put((self._generation, sentence))

    def wait(self) -> None:
        """Block until everything said so far has been spoken."""
        self._started_reply = False
        self._text.put((self._generation, _END))
        self._idle.wait()

    def stop(self) -> None:
        """Interrupt: drop anything queued and cut off the current sentence."""
        self._generation += 1
        self._pending = ""
        self._started_reply = False
        for q in (self._text, self._clips):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self.voice.stop()
        self.voice.release()
        self._finish()

    def _finish(self) -> None:
        if not self._idle.is_set():
            self._idle.set()
            self.on_idle()

    def _render_worker(self) -> None:
        while True:
            generation, item = self._text.get()
            if generation != self._generation:
                continue
            if item is not _END:
                try:
                    item = self.voice.render(str(item))
                except Exception as e:  # noqa: BLE001 - skip a sentence rather than kill the voice
                    print(f"[voice] couldn't speak a sentence: {e}", flush=True)
                    continue
            self._clips.put((generation, item))

    def _play_worker(self) -> None:
        speaking_gen = None  # generation of the reply currently being spoken
        while True:
            generation, clip = self._clips.get()
            if generation != self._generation:
                continue
            if clip is _END:
                speaking_gen = None
                self.voice.release()
                self._finish()
                continue
            if speaking_gen != generation:
                speaking_gen = generation
                self.on_start()
            self.voice.play(clip)

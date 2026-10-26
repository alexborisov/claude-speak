# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "kokoro-onnx>=0.4,<1",
#     "numpy>=2",
# ]
# ///
"""claude-speak TTS server: keeps Kokoro loaded and plays speech on request.

Listens for HTTP on a unix socket in the data dir. Speech from different
sessions is queued; a new /speak from the same session replaces its older one.
Exits by itself after a period with nothing to do.

Voices named clone:<name> are cloned from <data>/voices/<name>.wav with
Chatterbox Turbo. That needs mlx-audio (Apple Silicon), which the client adds
with `uv run --with mlx-audio` only when a clone voice is in use.

  GET  /health   model, player, voices and clone support
  POST /speak    {"text", "voice", "speed", "session", "label", "more"}; returns at once,
                 plays in turn. With "more": true, the rest follows through /append.
  POST /append   {"session", "text", "more"}; continue that session's /speak
  POST /warm     {"voice"}; play silence to wake the output device before speech is
                 ready, and load a cloned voice
  POST /synth    same body; returns the whole utterance as a WAV
  POST /stop     stop playback and clear the queue
  POST /shutdown stop and exit, so the client can restart with other dependencies
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import queue
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import wave
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODELS = {
    "fp32": "kokoro-v1.0.onnx",       # ~310 MB, best quality
    "fp16": "kokoro-v1.0.fp16.onnx",  # ~170 MB
    "int8": "kokoro-v1.0.int8.onnx",  # ~90 MB, fastest on slow CPUs
}
VOICES_FILE = "voices-v1.0.bin"

# Kokoro voice names start with a language code.
LANGS = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
         "i": "it", "j": "ja", "p": "pt-br", "z": "cmn"}

# Start speaking after the first chunk instead of waiting for the whole text.
# The first chunk is short so speech starts quickly; each later one may be up
# to 1.5 times as long as the one before, since it's synthesized while that
# one plays. Synthesis runs at about twice real time less a fixed cost per
# chunk, so doubling would leave audible gaps.
FIRST_CHUNK_CHARS = 50
CHUNK_CHARS = 250
MIN_CLAUSE_CHARS = 20

# How long a streaming /speak waits for its next /append before giving up.
STREAM_TIMEOUT = 20.0

# Log the timing of every chunk (CLAUDE_SPEAK_DEBUG=1), for tuning latency.
DEBUG = bool(os.environ.get("CLAUDE_SPEAK_DEBUG"))

# How long the output must be playing before speech starts, so the first words
# aren't lost while an idle device (Bluetooth especially) wakes up. Time already
# spent playing, e.g. silence from /warm while a summary is written, counts.
LEAD_IN = float(os.environ.get("CLAUDE_SPEAK_LEAD_IN", 2.0))
MIN_LEAD_IN = 0.25

# Longest /warm keeps the output awake while it waits for speech.
WARM_SECS = 20

# Output that went quiet less than this long ago is still awake. PipeWire
# suspends idle outputs after 5 seconds.
STAY_AWAKE = 3.0

# While another session has spoken this recently, speech starts with the name
# of the project it came from.
OTHERS_ACTIVE = 300.0

CLONE_PREFIX = "clone:"
CLONE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
CLONE_MODEL = os.environ.get("CLAUDE_SPEAK_CLONE_MODEL", "mlx-community/chatterbox-turbo-fp16")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def download(name: str, dest_dir: Path) -> Path:
    dest = dest_dir / name
    if dest.exists():
        return dest
    log(f"downloading {name} ...")
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{RELEASE}/{name}") as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    part.rename(dest)
    log(f"downloaded {name} ({dest.stat().st_size // (1 << 20)} MB)")
    return dest


def find_player() -> list[str] | None:
    """A command that plays a WAV file given as its last argument."""
    candidates = [
        ["afplay"],                                       # macOS
        ["pw-play"],                                      # PipeWire
        ["paplay"],                                       # PulseAudio
        ["aplay", "-q"],                                  # ALSA
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ]
    override = os.environ.get("CLAUDE_SPEAK_PLAYER")
    if override:
        candidates.insert(0, override.split())
    for cmd in candidates:
        if shutil.which(cmd[0]):
            return cmd
    return None


def chunk_limit(prev: int) -> int:
    """Longest next chunk, given the length of the one before (0 for the first)."""
    return min(CHUNK_CHARS, max(FIRST_CHUNK_CHARS, prev * 3 // 2))


# Where a long sentence can be cut: after punctuation, or failing that, before
# a conjunction, so each part still sounds like a phrase.
CLAUSE_BREAKS = (re.compile(r"[,;:—–]\s+"),
                 re.compile(r"\s+(?=(?:and|but|or|so|because|which|while|when|then|that|if)\s)"))


def _split_clause(sentence: str, limit: int) -> tuple[str, str]:
    """Cut a long sentence at its last clause break within limit or, if there is
    none, at its first break after it. Both parts stay MIN_CLAUSE_CHARS or longer."""
    cuts = [[m.end() for m in breaks.finditer(sentence)
             if MIN_CLAUSE_CHARS <= m.end() <= len(sentence) - MIN_CLAUSE_CHARS]
            for breaks in CLAUSE_BREAKS]
    within = [max(c for c in kind if c <= limit) for kind in cuts if any(c <= limit for c in kind)]
    beyond = [min(kind) for kind in cuts if kind]
    cut = (within or beyond or [0])[0]
    return (sentence[:cut].strip(), sentence[cut:].strip()) if cut else (sentence, "")


def split_chunks(text: str, prev: int = 0) -> list[str]:
    """Split text into chunks to synthesize one at a time. prev is the length of
    the utterance's chunk before this text, for text that arrives in pieces."""
    pending = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+|\n+", text.strip()) if s.strip()]
    chunks, current = [], ""
    while pending:
        sentence = pending.pop(0)
        limit = chunk_limit(len(chunks[-1]) if chunks else prev)
        if current and len(current) + 1 + len(sentence) > limit:
            if len(current) < MIN_CLAUSE_CHARS:
                # A fragment such as "project name:" leads the next clause instead.
                head, rest = _split_clause(sentence, limit - len(current) - 1)
                if rest:
                    chunks.append(f"{current} {head}")
                    current = ""
                    pending.insert(0, rest)
                    continue
            chunks.append(current)
            current = ""
            limit = chunk_limit(len(chunks[-1]))
        if not current and len(sentence) > limit:
            head, rest = _split_clause(sentence, limit)
            if rest:
                chunks.append(head)
                pending.insert(0, rest)
                continue
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class CloneEngine:
    """Speech in a cloned voice, from a short reference recording.

    The model loads on first use, so setups that only use Kokoro voices never
    pay for it. Speaker conditioning is computed once per reference file and
    reused until the file changes. Chatterbox has no speed control.

    All MLX work runs on one dedicated thread: it serializes cloned speech
    without holding up Kokoro during the first model download, and keeps MLX
    state on the thread that created it.
    """

    def __init__(self, voices_dir: Path):
        self.voices_dir = voices_dir
        self.available = importlib.util.find_spec("mlx_audio") is not None
        self.model = None
        self.conds: dict[str, tuple[float, object]] = {}  # voice -> (ref mtime, conditionals)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clone")

    def _ref(self, voice: str) -> Path | None:
        name = voice[len(CLONE_PREFIX):] if voice.startswith(CLONE_PREFIX) else ""
        return self.voices_dir / f"{name}.wav" if CLONE_NAME.fullmatch(name) else None

    def voices(self) -> list[str]:
        return sorted(CLONE_PREFIX + p.stem for p in self.voices_dir.glob("*.wav")
                      if CLONE_NAME.fullmatch(p.stem))

    def has(self, voice: str) -> bool:
        ref = self._ref(voice)
        return ref is not None and ref.exists()

    def synth(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        return self.worker.submit(self._synth, text, voice).result()

    def preload(self, voice: str) -> None:
        """Load the model and the voice in the background, ahead of speech."""
        if self.available and self.has(voice):
            self.worker.submit(self._prepare, voice)

    def _synth(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        self._prepare(voice)
        audio = [np.asarray(r.audio, dtype=np.float32) for r in self.model.generate(text)]
        return np.concatenate(audio), self.model.sample_rate

    def _prepare(self, voice: str) -> None:
        if self.model is None:
            # Progress bars would fill server.log on every utterance.
            os.environ.setdefault("TQDM_DISABLE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            from mlx_audio.tts.utils import load
            started = time.time()
            self.model = load(CLONE_MODEL)
            log(f"loaded {CLONE_MODEL} in {time.time() - started:.1f}s")
        ref = self._ref(voice)
        mtime = ref.stat().st_mtime
        cached = self.conds.get(voice)
        if not cached or cached[0] != mtime:
            self.model.prepare_conditionals(str(ref))
            self.conds[voice] = (mtime, self.model._conds)
        self.model._conds = self.conds[voice][1]


@dataclass
class Job:
    session: str
    text: str
    voice: str
    speed: float
    streaming: bool = False
    cancelled: bool = False
    closed: bool = False
    more: queue.Queue = field(default_factory=queue.Queue)  # appended text; None ends it

    def cancel(self) -> None:
        self.cancelled = True
        self.more.put(None)

    def pieces(self):
        yield self.text
        while self.streaming and not self.cancelled:
            try:
                piece = self.more.get(timeout=STREAM_TIMEOUT)
            except queue.Empty:
                log(f"no more text for {self.session[:8] or '-'} after {STREAM_TIMEOUT:g}s; finishing")
                return
            if piece is None:
                return
            yield piece


class Speaker:
    """Plays utterances one at a time, in order.

    Sessions take turns: speech from one session waits for another's to finish.
    A newer utterance from the same session replaces its older one, playing or
    queued, since that reply is out of date.
    """

    def __init__(self, kokoro: Kokoro, clone: CloneEngine, player: list[str] | None, silence: Path):
        self.kokoro = kokoro
        self.clone = clone
        self.player = player
        self.silence = silence
        self.voices = set(kokoro.get_voices())
        self.synth_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.wake = threading.Condition(self.state_lock)
        self.queue: list[Job] = []
        self.current: Job | None = None
        self.last_seen: dict[str, float] = {}  # session -> last /speak time
        self.proc: subprocess.Popen | None = None
        self.warm_proc: subprocess.Popen | None = None
        self.busy_until = 0.0
        # Output-awake tracking, guarded by state_lock.
        self.playing = 0
        self.awake_since = 0.0
        self.quiet_since = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def resolve_voice(self, voice: str) -> str:
        if voice in self.voices or (self.clone.available and self.clone.has(voice)):
            return voice
        if voice.startswith(CLONE_PREFIX):
            reason = "no reference recording" if self.clone.available else "mlx-audio not installed"
            log(f"can't use {voice} ({reason}); using af_heart")
        return "af_heart"

    def synth(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        voice = self.resolve_voice(voice)
        if voice.startswith(CLONE_PREFIX):
            try:
                return self.clone.synth(text, voice)
            except Exception as exc:
                log(f"cloned voice {voice} failed ({exc}); using af_heart")
                voice = "af_heart"
        with self.synth_lock:
            return self.kokoro.create(text, voice=voice, speed=speed,
                                      lang=LANGS.get(voice[:1], "en-us"))

    def idle(self) -> bool:
        with self.state_lock:
            return not self.current and not self.queue

    def stop(self) -> None:
        """Stop everything: what's playing and what's queued."""
        with self.state_lock:
            for job in self.queue:
                job.cancel()
            self.queue.clear()
            self._cancel_current()
            self._end_warm()
            self.busy_until = time.time()

    def _cancel_current(self) -> None:
        """Call with state_lock held."""
        if self.current:
            self.current.cancel()
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()

    def warm(self, voice: str = "") -> None:
        self.clone.preload(voice)
        with self.state_lock:
            if (not self.player or self.current or self.queue
                    or any(p and p.poll() is None for p in (self.proc, self.warm_proc))):
                return
            proc = self.warm_proc = self._spawn(self.silence)
        threading.Thread(target=self._reap, args=(proc,), daemon=True).start()

    def _end_warm(self) -> None:
        if self.warm_proc and self.warm_proc.poll() is None:
            self.warm_proc.terminate()

    def _spawn(self, path: Path | str) -> subprocess.Popen:
        """Start the player; call with state_lock held."""
        now = time.time()
        if self.playing == 0 and now - self.quiet_since > STAY_AWAKE:
            self.awake_since = now
        self.playing += 1
        return subprocess.Popen(self.player + [str(path)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _reap(self, proc: subprocess.Popen) -> None:
        proc.wait()
        with self.state_lock:
            self.playing -= 1
            if self.playing == 0:
                self.quiet_since = time.time()

    def lead_in(self) -> float:
        """Silence to put before speech so the output has been awake for LEAD_IN."""
        with self.state_lock:
            now = time.time()
            awake = self.playing > 0 or now - self.quiet_since <= STAY_AWAKE
            woke = now - self.awake_since if awake else 0.0
        return max(MIN_LEAD_IN, LEAD_IN - woke)

    def speak(self, text: str, voice: str, speed: float, session: str = "", label: str = "",
              more: bool = False) -> None:
        with self.state_lock:
            now = time.time()
            others = any(s != session and now - seen < OTHERS_ACTIVE for s, seen in self.last_seen.items())
            if session:
                self.last_seen[session] = now
            if label and others:
                text = f"{re.sub(r'[-_.]+', ' ', label)}: {text}"
            for job in self.queue:
                if job.session == session:
                    job.cancel()
            self.queue = [job for job in self.queue if not job.cancelled]
            if self.current and self.current.session == session:
                self._cancel_current()
            self.queue.append(Job(session, text, voice, speed, streaming=more))
            self.busy_until = now + 3600
            self.wake.notify()

    def append(self, session: str, text: str, more: bool) -> bool:
        """Continue the session's streaming utterance; False if there isn't one."""
        with self.state_lock:
            jobs = [j for j in [self.current, *self.queue]
                    if j and j.session == session and j.streaming and not j.closed and not j.cancelled]
            if not jobs:
                return False
            job = jobs[-1]
            if text:
                job.more.put(text)
            if not more:
                job.closed = True
                job.more.put(None)
            return True

    def _loop(self) -> None:
        while True:
            with self.state_lock:
                while not self.queue:
                    self.wake.wait()
                job = self.current = self.queue.pop(0)
            try:
                self._run(job)
            finally:
                with self.state_lock:
                    self.current = None
                    if not self.queue:
                        self.busy_until = time.time()

    def _run(self, job: Job) -> None:
        voice, speed = job.voice, job.speed
        if not self.player:
            log("no audio player found (tried afplay, pw-play, paplay, aplay, ffplay)")
            return
        clips: queue.Queue = queue.Queue(maxsize=2)
        prev = [0]  # length of the last chunk, so pieces that arrive later keep growing

        def produce():
            try:
                chunks = (chunk for piece in job.pieces()
                          for chunk in split_chunks(piece, prev=prev[0]))
                for i, chunk in enumerate(chunks):
                    if job.cancelled:
                        break
                    prev[0] = len(chunk)
                    started = time.time()
                    samples, rate = self.synth(chunk, voice, speed)
                    if DEBUG:
                        log(f"chunk {i}: {len(chunk)} chars in {time.time() - started:.2f}s "
                            f"-> {len(samples) / rate:.2f}s audio: {chunk!r}")
                    if i == 0:
                        lead = np.zeros(int(self.lead_in() * rate), samples.dtype)
                        samples = np.concatenate([lead, samples])
                    fd, path = tempfile.mkstemp(prefix="claude-speak-", suffix=".wav")
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(wav_bytes(samples, rate))
                    clips.put(path)
            except Exception as exc:
                log(f"synthesis failed: {exc}")
            finally:
                clips.put(None)

        threading.Thread(target=produce, daemon=True).start()
        while (path := clips.get()) is not None:
            try:
                with self.state_lock:
                    if job.cancelled:
                        continue
                    self._end_warm()
                    proc = self.proc = self._spawn(path)
                self._reap(proc)
            finally:
                os.unlink(path)


class Handler(BaseHTTPRequestHandler):
    server: "Server"

    def log_message(self, *_):  # unix sockets have no client address to log
        pass

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        self.server.touch()
        if self.path == "/health":
            speaker = self.server.speaker
            self._json(200, {"ok": True, "model": self.server.model_name,
                             "player": " ".join(speaker.player or ["none"]),
                             "voices": sorted(speaker.voices),
                             "clone": speaker.clone.available,
                             "clone_voices": speaker.clone.voices()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        self.server.touch()
        speaker = self.server.speaker
        try:
            body = self._body()
            text = str(body.get("text", "")).strip()
            voice = str(body.get("voice") or "af_heart")
            speed = min(2.0, max(0.5, float(body.get("speed") or 1.0)))
            if self.path == "/stop":
                speaker.stop()
                self._json(200, {"ok": True})
            elif self.path == "/shutdown":
                if not speaker.idle():
                    self._json(409, {"error": "busy speaking"})
                    return
                self._json(200, {"ok": True})
                log("shutdown requested; exiting")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            elif self.path == "/warm":
                speaker.warm(str(body.get("voice") or ""))
                self._json(202, {"ok": True})
            elif self.path == "/speak" and text:
                session = str(body.get("session") or "")
                log(f"speak [{voice} x{speed}] {session[:8] or '-'}: {text[:100]!r}{'...' if len(text) > 100 else ''}")
                speaker.speak(text, voice, speed, session, str(body.get("label") or ""),
                              more=bool(body.get("more")))
                self._json(202, {"ok": True})
            elif self.path == "/append":
                if speaker.append(str(body.get("session") or ""), text, bool(body.get("more"))):
                    self._json(202, {"ok": True})
                else:
                    self._json(404, {"error": "no open /speak for that session"})
            elif self.path == "/synth" and text:
                samples, rate = speaker.synth(text, voice, speed)
                payload = wav_bytes(samples, rate)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._json(400, {"error": "expected /speak or /synth with text, /append, /warm, /stop or /shutdown"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, speaker: Speaker, model_name: str):
        self.speaker = speaker
        self.model_name = model_name
        self.last_used = time.time()
        super().__init__(path, Handler)

    def touch(self) -> None:
        self.last_used = time.time()


def socket_in_use(path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(path))
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", default=os.environ.get("CLAUDE_SPEAK_MODEL", "fp32"), choices=MODELS)
    parser.add_argument("--idle-timeout", type=float,
                        default=float(os.environ.get("CLAUDE_SPEAK_IDLE_TIMEOUT", 1800)),
                        help="seconds without requests before the server exits")
    args = parser.parse_args()

    data = Path(args.data_dir).expanduser()
    sock = Path(args.socket)
    if socket_in_use(sock):
        log("already running")
        return
    sock.unlink(missing_ok=True)

    models = data / "models"
    models.mkdir(parents=True, exist_ok=True)
    model_file = download(MODELS[args.model], models)
    voices_file = download(VOICES_FILE, models)

    started = time.time()
    kokoro = Kokoro(str(model_file), str(voices_file))
    silence = data / "silence.wav"
    silence.write_bytes(wav_bytes(np.zeros(WARM_SECS * 24000, np.float32), 24000))
    clone = CloneEngine(data / "voices")
    speaker = Speaker(kokoro, clone, find_player(), silence)
    speaker.synth("Ready.", "af_heart", 1.0)  # warm up before accepting requests
    log(f"loaded {MODELS[args.model]} in {time.time() - started:.1f}s; "
        f"player: {' '.join(speaker.player or ['none'])}; "
        f"cloning: {'available' if clone.available else 'off'}")

    server = Server(str(sock), speaker, MODELS[args.model])

    def idle_watch():
        while True:
            time.sleep(15)
            now = time.time()
            if now - server.last_used > args.idle_timeout and now > speaker.busy_until:
                log("idle; exiting")
                server.shutdown()
                return

    threading.Thread(target=idle_watch, daemon=True).start()
    log(f"listening on {sock}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())

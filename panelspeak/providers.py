"""Provider interfaces for the model-driven stages.

The pipeline talks to three kinds of model: a vision-language model (stage 2),
a chat model (stage 3 + voice classification), and a text-to-speech engine
(stage 4). Today each script hard-codes an Ollama URL and a Piper path. These
ABCs are the seam that lets the homelab build point at whatever the user runs
(Ollama, LM Studio, llama.cpp, vLLM for the LLMs; Piper, Chatterbox, VibeVoice,
Orpheus for TTS) and lets the tests run against fakes.

Only the interface lives here -- concrete adapters live in
``panelspeak/adapters/`` and are added as they're built. Contract tests in
``tests/providers/`` pin the behaviour every adapter must honour.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class TTSRequest:
    text: str
    #: logical speaker name; the engine maps it to a voice / reference clip.
    speaker: str = "NARRATOR"
    #: path to a reference audio clip for zero-shot voice cloning, if the
    #: engine supports it (Chatterbox, VibeVoice, XTTS).
    reference_audio: str | None = None
    #: free-text emotion hint from stage 3 ("panicked", "weary", ...).
    emotion: str | None = None
    #: 0..1; engines that expose an exaggeration / intensity knob use this.
    intensity: float = 0.0
    #: the line before and after, for engines that use context for prosody
    #: (Higgs, VibeVoice). Empty string when not available.
    context_before: str = ""
    context_after: str = ""


@dataclass
class TTSResult:
    #: 16-bit PCM WAV bytes.
    audio: bytes
    sample_rate: int
    duration_s: float
    voice: str = ""


@dataclass
class PanelAnalysis:
    text: str
    error: str | None = None
    raw: dict = field(default_factory=dict)


class VisionModel(abc.ABC):
    """Describes a comic panel: scene, characters, transcribed text."""

    @abc.abstractmethod
    def analyze_panel(self, image_path: str, *, prompt: str) -> PanelAnalysis:
        ...

    @abc.abstractmethod
    def health(self) -> bool:
        """Cheap reachability + capability check for the setup wizard."""


class ChatModel(abc.ABC):
    """Plain text-in / text-out completion."""

    @abc.abstractmethod
    def complete(self, prompt: str, *, max_tokens: int = 2000, num_ctx: int = 16384) -> str:
        ...

    @abc.abstractmethod
    def health(self) -> bool:
        ...


class TTSProvider(abc.ABC):
    """Turns one line into audio."""

    #: whether ``TTSRequest.reference_audio`` is honoured.
    supports_voice_cloning: bool = False
    #: whether ``TTSRequest.emotion`` / ``intensity`` do anything.
    supports_emotion: bool = False
    #: whether ``context_before`` / ``context_after`` influence prosody.
    supports_context: bool = False
    #: inline non-verbal tags the engine understands ("<sigh>", ...).
    nonverbal_tags: frozenset[str] = frozenset()

    @abc.abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResult:
        ...

    @abc.abstractmethod
    def health(self) -> bool:
        ...

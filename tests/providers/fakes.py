"""In-memory fake providers, used by the contract tests and available for any
other test that needs a model without a model.
"""

from __future__ import annotations

import struct

from panelspeak.providers import (
    ChatModel,
    PanelAnalysis,
    TTSProvider,
    TTSRequest,
    TTSResult,
    VisionModel,
)


def _silence_wav(duration_s: float, sample_rate: int = 22050) -> tuple[bytes, int]:
    n = int(duration_s * sample_rate)
    return struct.pack(f"<{n}h", *([0] * n)), sample_rate


class FakeVisionModel(VisionModel):
    def __init__(self, canned: str = "A quiet room.\nNARRATOR: It was over."):
        self.canned = canned
        self.calls: list[str] = []
        self._healthy = True

    def analyze_panel(self, image_path: str, *, prompt: str) -> PanelAnalysis:
        self.calls.append(image_path)
        if not self._healthy:
            return PanelAnalysis(text="", error="model offline")
        return PanelAnalysis(text=self.canned, raw={"eval_count": 42})

    def health(self) -> bool:
        return self._healthy


class FakeChatModel(ChatModel):
    def __init__(self, responder=None):
        self.responder = responder or (lambda p: "NARRATOR: Something happened.")
        self.prompts: list[str] = []
        self._healthy = True

    def complete(self, prompt: str, *, max_tokens: int = 2000, num_ctx: int = 16384) -> str:
        self.prompts.append(prompt)
        return self.responder(prompt)

    def health(self) -> bool:
        return self._healthy


class FakeTTS(TTSProvider):
    supports_voice_cloning = True
    supports_emotion = True
    supports_context = True
    nonverbal_tags = frozenset({"<sigh>", "<laugh>", "<gasp>"})

    def __init__(self, seconds_per_char: float = 0.05):
        self.seconds_per_char = seconds_per_char
        self.requests: list[TTSRequest] = []
        self._healthy = True

    def synthesize(self, request: TTSRequest) -> TTSResult:
        self.requests.append(request)
        dur = max(0.1, len(request.text) * self.seconds_per_char)
        audio, sr = _silence_wav(dur)
        return TTSResult(audio=audio, sample_rate=sr, duration_s=dur,
                         voice=request.speaker.lower())

    def health(self) -> bool:
        return self._healthy

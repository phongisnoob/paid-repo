"""Frozen speech encoders -> frame features [T, 768] at 50 Hz (20 ms/frame).

!! These wrappers need `transformers` + downloaded weights and have NOT been run in the
author's sandbox (no access to the HF hub there). Everything downstream of them
(paid/pooling.py etc.) is unit-tested with a fake encoder. Check the printed shapes on the
first utterance: T should be ~ duration/0.02 and D = 768.
"""
import math

import torch

SR = 16000


class WhisperSmallEncoder:
    frame_dt = 0.02

    def __init__(self, device="cuda", fp16=True, name="openai/whisper-small"):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.device, self.dtype = device, (torch.float16 if fp16 else torch.float32)
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.model = WhisperModel.from_pretrained(name).get_encoder().to(device, self.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: 1-D float tensor at 16 kHz -> [T, 768] float32 on CPU (real length only)."""
        n = wav.shape[0]
        feats = self.fe(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_features
        out = self.model(feats.to(self.device, self.dtype)).last_hidden_state[0]   # [1500, 768]
        t_real = min(out.shape[0], math.ceil(min(n, 30 * SR) / 320))   # drop the 30 s padding
        return out[:t_real].float().cpu()


class WavLMBasePlusEncoder:
    frame_dt = 0.02

    def __init__(self, device="cuda", fp16=True, name="microsoft/wavlm-base-plus"):
        from transformers import AutoFeatureExtractor, WavLMModel
        self.device, self.dtype = device, (torch.float16 if fp16 else torch.float32)
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = WavLMModel.from_pretrained(name).to(device, self.dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.fe(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_values
        return self.model(x.to(self.device, self.dtype)).last_hidden_state[0].float().cpu()


class FakeEncoder:
    """Test double: frame t = vector filled with the value t (20 ms hop, no model needed).
    Lets the whole wav + TextGrid -> cached-record path be tested without HF weights."""
    frame_dt = 0.02

    def __init__(self, device="cpu", fp16=False, D=8):
        self.D = D

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        T = math.ceil(wav.shape[0] / 320)
        return torch.arange(T, dtype=torch.float32).unsqueeze(1).repeat(1, self.D)


ENCODERS = {"whisper": WhisperSmallEncoder, "wavlm": WavLMBasePlusEncoder, "fake": FakeEncoder}

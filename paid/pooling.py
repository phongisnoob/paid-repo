"""Frame -> phoneme-span pooling (Eqs. 3-4 of the paper), plus the utterance-level
statistics needed by the baselines. Pure torch, no encoder dependency => unit-testable.

Deviations / decisions the paper leaves open (all documented here on purpose):
  * Frame timestamp tau_t is the frame CENTRE: (t + 0.5) * frame_dt.
  * Spans are half-open [s, e) instead of the closed [s, e] of Eq. (3), so a frame sitting
    exactly on a shared boundary is not counted twice (MFA boundaries are multiples of
    10 ms, frame centres of a 20 ms encoder are odd multiples of 10 ms -> ties are common).
  * A span that contains no frame centre (shorter than one frame) would make Eq. (4) divide
    by zero. We use the single frame closest to the span midpoint instead.
"""
import torch

FRAME_DT = 0.02  # Whisper-small encoder and WavLM Base+ both output 50 Hz


def _round6(x):
    return torch.round(x * 1e6) / 1e6


def span_pool(frames: torch.Tensor, spans, frame_dt: float = FRAME_DT):
    """frames [T, D]; spans list of (start_sec, end_sec).

    Returns tokens [N, D] (float32), speech_mask [T] (bool: frames used by >=1 span),
    n_fallback (#spans that had no frame centre inside).
    """
    T = frames.shape[0]
    tau = _round6((torch.arange(T, dtype=torch.float64) + 0.5) * frame_dt)
    speech = torch.zeros(T, dtype=torch.bool)
    tokens, n_fb = [], 0
    frames = frames.float()
    for s, e in spans:
        s6, e6 = round(s, 6), round(e, 6)
        sel = (tau >= s6) & (tau < e6)
        if sel.any():
            tokens.append(frames[sel].mean(0))
            speech |= sel
        else:
            c = int(round(0.5 * (s6 + e6) / frame_dt - 0.5))
            c = min(max(c, 0), T - 1)
            tokens.append(frames[c])
            speech[c] = True
            n_fb += 1
    return torch.stack(tokens), speech, n_fb


def build_record(frames: torch.Tensor, alignment, frame_dt: float = FRAME_DT,
                 keep_frames: bool = False):
    """alignment: [(phone_id, start, end)]. Returns a dict (tensors stored in fp16)."""
    if len(alignment) == 0:
        raise ValueError("empty alignment")
    spans = [(s, e) for _, s, e in alignment]
    tokens, speech, n_fb = span_pool(frames, spans, frame_dt)
    f32 = frames.float()
    rec = {
        "tokens": tokens.half(),
        "phones": torch.tensor([p for p, _, _ in alignment], dtype=torch.int16),
        "utt_mean": f32.mean(0).half(),                 # all (real-length) frames
        "speech_mean": f32[speech].mean(0).half(),      # only frames inside phoneme spans
        "n_frames": int(frames.shape[0]),
        "n_fallback": n_fb,
    }
    if keep_frames:
        rec["frames"] = frames.half()
    return rec

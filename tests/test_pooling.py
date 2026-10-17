import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paid.pooling import build_record, span_pool  # noqa: E402

DT = 0.02


def make_frames(T=50, D=4):
    # frame t has constant value t in every dim  => pooled value = mean of frame indices
    return torch.arange(T, dtype=torch.float32).unsqueeze(1).repeat(1, D)


def test_span_mean_matches_manual():
    f = make_frames()
    # centres: (t+.5)*0.02 ; span [0.10, 0.30) -> t where 0.10 <= (t+.5)*.02 < 0.30
    # => t = 5..14 (centres 0.11 ... 0.29)  -> mean index = 9.5
    tok, speech, nfb = span_pool(f, [(0.10, 0.30)], DT)
    assert nfb == 0
    assert torch.allclose(tok[0], torch.full((4,), 9.5))
    assert speech.nonzero().flatten().tolist() == list(range(5, 15))


def test_adjacent_spans_do_not_share_frames_on_boundary():
    # boundary at 0.10 coincides with the centre of frame 4 (0.09)? no -> use 0.11 = centre of t=5
    f = make_frames()
    tok, speech, _ = span_pool(f, [(0.0, 0.11), (0.11, 0.30)], DT)
    # half-open: frame 5 (centre 0.11) belongs ONLY to the second span
    assert torch.allclose(tok[0], torch.full((4,), 2.0))          # frames 0..4
    assert torch.allclose(tok[1], torch.full((4,), 9.5))          # frames 5..14


def test_sub_frame_span_uses_nearest_frame_instead_of_nan():
    f = make_frames()
    tok, speech, nfb = span_pool(f, [(0.101, 0.105)], DT)   # contains no frame centre
    assert nfb == 1 and not torch.isnan(tok).any()
    assert torch.allclose(tok[0], torch.full((4,), 5.0))    # centre 0.103 -> frame 5 (0.11)
    assert speech[5]


def test_span_past_end_of_audio_is_clamped():
    f = make_frames(T=10)
    tok, _, nfb = span_pool(f, [(5.0, 5.1)], DT)
    assert nfb == 1 and torch.allclose(tok[0], torch.full((4,), 9.0))


def test_build_record_stats():
    f = make_frames()
    ali = [(3, 0.10, 0.30), (7, 0.40, 0.60)]
    r = build_record(f, ali, DT, keep_frames=True)
    assert r["tokens"].shape == (2, 4) and r["tokens"].dtype == torch.float16
    assert r["phones"].tolist() == [3, 7]
    assert torch.allclose(r["utt_mean"].float(), torch.full((4,), 24.5))
    used = list(range(5, 15)) + list(range(20, 30))
    assert torch.allclose(r["speech_mean"].float(), torch.full((4,), sum(used) / len(used)))
    assert r["frames"].shape == (50, 4)

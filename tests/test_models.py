import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paid.data import collate  # noqa: E402
from paid.models import (PAIDModel, build_model, masked_mean,  # noqa: E402
                         shuffle_ids_within_utt, VARIANTS)


def fake_records(lengths, D=768, with_frames=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    recs = []
    for i, n in enumerate(lengths):
        r = {"tokens": torch.randn(n, D, generator=g).half(),
             "phones": torch.randint(0, 39, (n,), generator=g).to(torch.int16),
             "utt_mean": torch.randn(D, generator=g).half(),
             "speech_mean": torch.randn(D, generator=g).half(),
             "spk": "ABA", "y": i % 6, "utt": f"u{i}"}
        if with_frames:
            r["frames"] = torch.randn(n * 3, D, generator=g).half()
        recs.append(r)
    return recs


def test_faithful_param_count_matches_paper_description():
    m = build_model("paid")
    # Linear 768->192 (+bias) + LN(192) + Emb 39x64 + Linear 256->6 (+bias)
    expected = 768 * 192 + 192 + 2 * 192 + 39 * 64 + 256 * 6 + 6
    assert sum(p.numel() for p in m.parameters()) == expected == 152_070


@pytest.mark.parametrize("name", [v for v in VARIANTS if v != "frameattn"])
def test_forward_shapes_and_backward(name):
    batch = collate(fake_records([5, 9, 3]))
    m = build_model(name)
    out = m(batch)
    assert out.shape == (3, 6) and torch.isfinite(out).all()
    out.sum().backward()


def test_frameattn_shapes():
    batch = collate(fake_records([5, 9, 3], with_frames=True))
    out = build_model("frameattn")(batch)
    assert out.shape == (3, 6) and torch.isfinite(out).all()


@pytest.mark.parametrize("fusion", ["concat_linear", "mlp", "transformer"])
def test_padding_does_not_change_outputs(fusion):
    """An utterance must get the same logits alone or batched with a much longer one."""
    short, long_ = fake_records([4, 30])
    m = PAIDModel(fusion=fusion).eval()
    alone = m(collate([short]))
    batched = m(collate([short, long_]))[0:1]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_masked_mean_ignores_padding():
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert masked_mean(x, mask).item() == 2.0


def test_shuffle_keeps_multiset_of_valid_ids_and_changes_order():
    torch.manual_seed(0)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 0, 0, 0, 0, 0]])
    mask = torch.tensor([[True] * 8, [True] * 3 + [False] * 5])
    out = shuffle_ids_within_utt(ids, mask)
    assert sorted(out[0].tolist()) == sorted(ids[0].tolist())
    assert sorted(out[1, :3].tolist()) == [9, 10, 11]
    assert not torch.equal(out[0], ids[0])


def test_concat_linear_is_additive_no_interaction():
    """Documents the architectural point: with mean-pool + linear head, logits are
    f(mean acoustics) + g(mean phone embedding) -> permuting phone ids among tokens changes
    NOTHING (bag-of-phonemes). The 'shuffle' control must therefore be a no-op for the
    faithful model and only matter for models with token-level interaction."""
    recs = fake_records([12, 7])
    batch = collate(recs)
    torch.manual_seed(0)
    faithful = PAIDModel(shuffle_phone_id=False).eval()
    shuffled = PAIDModel(shuffle_phone_id=True).eval()
    shuffled.load_state_dict(faithful.state_dict())
    assert torch.allclose(faithful(batch), shuffled(batch), atol=1e-5)

    mlp = PAIDModel(fusion="mlp").eval()
    mlp_s = PAIDModel(fusion="mlp", shuffle_phone_id=True).eval()
    mlp_s.load_state_dict(mlp.state_dict())
    assert not torch.allclose(mlp(batch), mlp_s(batch), atol=1e-4)

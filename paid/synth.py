"""Synthetic L2-ARCTIC-shaped features for smoke tests (no audio, no encoder needed).

mode='interaction': classes come in 3 pairs (2p, 2p+1). Each pair has two directions u_p, v_p.
    class 2p  : tokens of phone-group A are shifted by +a*u_p, group B by +a*v_p
    class 2p+1: tokens of phone-group A are shifted by +a*v_p, group B by +a*u_p
  Both classes of a pair contain exactly the same bag of acoustic shifts and the same bag
  of phone ids; they differ ONLY in which acoustic shift goes with which phone group.
  => any model that sees acoustics and phone ids separately (bag-of-tokens, e.g. the
     faithful PAID with mean-pool + linear) is capped at 50% (pair known, member guessed);
  => a model that combines each token's acoustics with ITS OWN phone id can reach ~100%.
mode='mean': class evidence is a plain mean shift (solvable by any pooling model).
"""
import os
import zlib

import numpy as np
import torch

from .l2arctic import ALL_SPEAKERS, spk_label


def make_speaker(spk, n_utt, D=32, mode="interaction", a=2.0, seed=0, n_classes=6):
    y = spk_label(spk)
    rng = np.random.default_rng(seed)                                   # shared across spk
    dirs = np.random.default_rng(1234).normal(size=(n_classes, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    spk_rng = np.random.default_rng(zlib.crc32(spk.encode()) + seed)
    spk_off = spk_rng.normal(scale=0.15, size=D)                         # speaker identity
    recs = []
    for i in range(n_utt):
        n = 2 * int(spk_rng.integers(6, 11))
        group = np.array([1] * (n // 2) + [-1] * (n // 2))
        spk_rng.shuffle(group)
        phones = np.where(group == 1, spk_rng.integers(0, 20, n), spk_rng.integers(20, 39, n))
        x = spk_rng.normal(size=(n, D)) + spk_off
        if mode == "interaction":
            u, v = dirs[2 * (y // 2)], dirs[2 * (y // 2) + 1]
            if y % 2 == 1:
                u, v = v, u
            x += a * np.where(group[:, None] == 1, u[None, :], v[None, :])
        else:
            x += a * dirs[y]
        mean = x.mean(0)
        recs.append({"tokens": torch.tensor(x).half(), "phones": torch.tensor(phones, dtype=torch.int16),
                     "utt_mean": torch.tensor(mean).half(), "speech_mean": torch.tensor(mean).half(),
                     "n_frames": 3 * n, "n_fallback": 0, "utt": f"{spk}_{i:04d}"})
    return recs


def write_fake_feature_dir(out_dir, n_utt=25, **kw):
    os.makedirs(out_dir, exist_ok=True)
    for spk in ALL_SPEAKERS:
        torch.save(make_speaker(spk, n_utt, **kw), os.path.join(out_dir, f"{spk}.pt"))

import os
import subprocess
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from paid.engine import train_fold  # noqa: E402
from paid.l2arctic import ALL_SPEAKERS, fold_split, spk_label  # noqa: E402
from paid.models import build_model  # noqa: E402
from paid.synth import make_speaker, write_fake_feature_dir  # noqa: E402


def _fold0(mode):
    by = {}
    for s in ALL_SPEAKERS:
        recs = make_speaker(s, 25, D=32, mode=mode)
        for r in recs:
            r["spk"], r["y"] = s, spk_label(s)
        by[s] = recs
    tr, te = fold_split(0)
    return [r for s in tr for r in by[s]], [r for s in te for r in by[s]]


def _acc(name, train, test):
    torch.manual_seed(0)
    m = build_model(name, in_dim=32)
    _, met, _ = train_fold(m, train, test, "cpu", epochs=15, lr=1e-3, batch_size=16,
                           log_fn=lambda *_: None)
    return met["acc"]


def test_plain_mean_shift_is_learned_by_faithful_model():
    train, test = _fold0("mean")
    assert _acc("paid", train, test) > 90


def test_phone_acoustic_pairing_needs_token_level_interaction():
    """Classes differ ONLY in which acoustic shift pairs with which phone group.
    Faithful PAID (mean-pool + linear) cannot see that; an MLP with phone ids can;
    removing / shuffling the ids destroys it (=> the id branch is really being used)."""
    train, test = _fold0("interaction")
    faithful = _acc("paid", train, test)
    mlp = _acc("paid_mlp", train, test)
    mlp_nopid = _acc("paid_mlp_nopid", train, test)
    mlp_shuf = _acc("paid_mlp_shuffle_pid", train, test)
    assert faithful < 65 and mlp_nopid < 65 and mlp_shuf < 65     # ~50% cap (3 pairs)
    assert mlp > 80
    assert mlp > faithful + 25


def test_cli_pipeline_train_and_report(tmp_path):
    feats, runs = str(tmp_path / "feats"), str(tmp_path / "runs")
    write_fake_feature_dir(feats, n_utt=12, D=32, mode="mean")
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                        "--feat_dir", feats, "--out", runs, "--variants", "paid", "utt_proj",
                        "--folds", "0", "1", "--epochs", "3", "--in_dim", "32",
                        "--device", "cpu"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(os.path.join(runs, "paid", "fold1", "preds.csv"))
    rep = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "report.py"), runs],
                         capture_output=True, text=True, env=env)
    assert rep.returncode == 0, rep.stderr[-2000:]
    assert "paid" in rep.stdout and "utt_proj" in rep.stdout
    cmp_ = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "report.py"), runs,
                           "--compare", "paid", "utt_proj"], capture_output=True, text=True,
                          env=env)
    assert cmp_.returncode == 0, cmp_.stderr[-2000:]
    assert "95% CI" in cmp_.stdout

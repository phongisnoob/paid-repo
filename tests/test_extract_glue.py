"""wav + MFA-style TextGrid -> cached records, using the fake encoder (frame t == value t)."""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from paid.data import collate, load_speaker  # noqa: E402
from paid.phones import PHONE2ID  # noqa: E402
from tests.test_data_protocol import TG  # noqa: E402


def test_extract_script_with_fake_encoder(tmp_path):
    wav_root, mfa_out, out = tmp_path / "wav_root", tmp_path / "mfa", tmp_path / "feats"
    (wav_root / "ABA" / "wav").mkdir(parents=True)
    (mfa_out / "ABA").mkdir(parents=True)
    sf.write(str(wav_root / "ABA" / "wav" / "arctic_a0001.wav"),
             np.zeros(16000, dtype=np.float32), 16000)                  # 1.0 s -> 50 frames
    (mfa_out / "ABA" / "arctic_a0001.TextGrid").write_text(TG)           # DH .5-.6, IH .6-.8, S .9-1.0
    # an utterance MFA failed on: wav exists, TextGrid has no phone intervals -> must be skipped
    sf.write(str(wav_root / "ABA" / "wav" / "arctic_a0002.wav"), np.zeros(16000, np.float32), 16000)
    (mfa_out / "ABA" / "arctic_a0002.TextGrid").write_text(TG.replace('text = "DH"', 'text = ""')
                                                           .replace('text = "IH1"', 'text = ""')
                                                           .replace('text = "S"', 'text = ""'))
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "extract_features.py"),
                        "--enc", "fake", "--wav_root", str(wav_root), "--mfa_out", str(mfa_out),
                        "--out", str(out), "--speakers", "ABA", "--device", "cpu", "--keep_frames"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "1 utts, 1 skipped" in r.stdout, r.stdout
    recs = load_speaker(str(out), "ABA")
    assert len(recs) == 1 and recs[0]["utt"] == "arctic_a0001"
    rec = recs[0]
    # spans: [.5,.6) -> frames 25..29 -> 27 ; [.6,.8) -> 30..39 -> 34.5 ; [.9,1.0) -> 45..49 -> 47
    assert rec["tokens"][:, 0].float().tolist() == [27.0, 34.5, 47.0]
    assert rec["phones"].tolist() == [PHONE2ID["DH"], PHONE2ID["IH"], PHONE2ID["S"]]
    assert rec["frames"].shape[0] == 50
    b = collate(recs)
    assert b["tokens"].shape == (1, 3, 8) and b["mask"].all()

"""Run the frozen encoder once per utterance and cache everything the models need.

    python scripts/extract_features.py --enc whisper --wav_root /data/L2-ARCTIC \
        --mfa_out mfa_out --out feats/whisper
    python scripts/extract_features.py --enc wavlm   ... --out feats/wavlm [--keep_frames]

Per speaker -> <out>/<SPK>.pt  (list of records, see paid/pooling.build_record).
Disk: pooled tokens only ~ 40 tokens x 768 x fp16 per utterance (~ 1.5 GB in total).
--keep_frames additionally stores every frame (~ 7 GB fp16) and is only needed for the
'frameattn' baseline.  Re-running skips speakers whose .pt already exists.
"""
import argparse
import os
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paid.encoders import ENCODERS, SR  # noqa: E402
from paid.l2arctic import ALL_SPEAKERS  # noqa: E402
from paid.pooling import build_record  # noqa: E402
from paid.textgrid import read_alignment  # noqa: E402


def load_wav16k(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != SR:
        import torchaudio.functional as AF   # only needed if audio is not already 16 kHz
        wav = AF.resample(wav, sr, SR)
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--wav_root", required=True, help="L2-ARCTIC root (<SPK>/wav/*.wav)")
    ap.add_argument("--mfa_out", required=True, help="MFA output dir (<SPK>/<utt>.TextGrid)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--speakers", nargs="+", default=ALL_SPEAKERS)
    ap.add_argument("--keep_frames", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="run encoder in fp32 (default fp16)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    enc = ENCODERS[args.enc](device=args.device, fp16=not args.fp32)
    first = True
    for spk in args.speakers:
        dst = os.path.join(args.out, f"{spk}.pt")
        if os.path.exists(dst):
            print(f"{spk}: exists, skipped")
            continue
        grid_dir = os.path.join(args.mfa_out, spk)
        if not os.path.isdir(grid_dir):
            print(f"[WARN] {spk}: no TextGrids in {grid_dir}")
            continue
        recs, n_bad, n_fb, n_tok, t0 = [], 0, 0, 0, time.time()
        for fn in sorted(os.listdir(grid_dir)):
            if not fn.endswith(".TextGrid"):
                continue
            utt = fn[: -len(".TextGrid")]
            wav_path = os.path.join(args.wav_root, spk, "wav", utt + ".wav")
            try:
                ali = read_alignment(os.path.join(grid_dir, fn))
                if not ali:
                    raise ValueError("no valid phone intervals")
                frames = enc(load_wav16k(wav_path))
                if first:
                    print(f"first utterance: frames {tuple(frames.shape)}, "
                          f"audio->{frames.shape[0] * enc.frame_dt:.2f}s, "
                          f"{len(ali)} phones, last phone ends at {ali[-1][2]:.2f}s")
                    first = False
                rec = build_record(frames, ali, enc.frame_dt, keep_frames=args.keep_frames)
            except Exception as ex:  # keep going, but never silently
                n_bad += 1
                print(f"  [skip] {spk}/{utt}: {ex}")
                continue
            rec["utt"] = utt
            n_fb += rec["n_fallback"]
            n_tok += rec["tokens"].shape[0]
            recs.append(rec)
        torch.save(recs, dst)
        print(f"{spk}: {len(recs)} utts, {n_bad} skipped, {n_tok} tokens "
              f"({n_fb} sub-frame spans -> nearest-frame fallback), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

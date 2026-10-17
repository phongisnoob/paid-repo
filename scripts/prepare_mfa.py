"""Build an MFA-ready corpus from the L2-ARCTIC release.

L2-ARCTIC layout (per speaker):  <root>/<SPK>/wav/*.wav, <root>/<SPK>/transcript/*.txt
MFA wants:                       <out>/<SPK>/<utt>.wav + <utt>.lab (same basename)

We symlink wavs and copy the prompt text (lower-cased, whitespace-normalised) to .lab.
Then run (MFA 3.x, conda env):
    mfa model download acoustic    english_us_arpa
    mfa model download dictionary  english_us_arpa
    mfa align <out> english_us_arpa english_us_arpa <mfa_out> -j 8 --clean \
        --beam 100 --retry_beam 400
(--beam/--retry_beam are enlarged because non-native speech otherwise makes more utterances
fail to align; failed utterances are simply missing from <mfa_out> and are skipped, with a
count, by extract_features.py.)
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paid.l2arctic import ALL_SPEAKERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="L2-ARCTIC root containing <SPK>/ folders")
    ap.add_argument("--out", required=True, help="MFA corpus dir to create")
    ap.add_argument("--copy_wav", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    total = 0
    for spk in ALL_SPEAKERS:
        wav_dir = os.path.join(args.root, spk, "wav")
        txt_dir = os.path.join(args.root, spk, "transcript")
        if not (os.path.isdir(wav_dir) and os.path.isdir(txt_dir)):
            print(f"[WARN] {spk}: expected {wav_dir} and {txt_dir} - skipped")
            continue
        out_dir = os.path.join(args.out, spk)
        os.makedirs(out_dir, exist_ok=True)
        n, missing = 0, 0
        for fn in sorted(os.listdir(wav_dir)):
            if not fn.lower().endswith(".wav"):
                continue
            utt = os.path.splitext(fn)[0]
            txt = os.path.join(txt_dir, utt + ".txt")
            if not os.path.exists(txt):
                missing += 1
                continue
            with open(txt, encoding="utf-8", errors="ignore") as f:
                text = re.sub(r"\s+", " ", f.read()).strip().lower()
            if not text:
                missing += 1
                continue
            dst = os.path.join(out_dir, fn)
            if not os.path.exists(dst):
                if args.copy_wav:
                    import shutil
                    shutil.copy2(os.path.join(wav_dir, fn), dst)
                else:
                    os.symlink(os.path.abspath(os.path.join(wav_dir, fn)), dst)
            with open(os.path.join(out_dir, utt + ".lab"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            n += 1
        total += n
        print(f"{spk}: {n} utterances prepared, {missing} without transcript")
    print(f"total {total} utterances -> {args.out}")


if __name__ == "__main__":
    main()

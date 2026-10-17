"""Run the 4-fold speaker-disjoint protocol for one or more model variants.

Examples
  # faithful reproduction on Whisper features
  python scripts/train.py --feat_dir feats/whisper --out runs/whisper --variants paid
  # everything (paper ablations + the controls the paper lacks)
  python scripts/train.py --feat_dir feats/whisper --out runs/whisper \
      --variants paid paid_nopid utt utt_proj speechutt_proj paid_shuffle_pid paid_mlp paid_tf
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paid.data import load_speakers  # noqa: E402
from paid.engine import train_fold  # noqa: E402
from paid.l2arctic import ALL_SPEAKERS, N_FOLDS, fold_split  # noqa: E402
from paid.models import VARIANTS, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["paid"], choices=sorted(VARIANTS))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--in_dim", type=int, default=768)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_test_curve", action="store_true",
                    help="print test metrics every epoch (diagnostic only)")
    args = ap.parse_args()

    print(f"loading features from {args.feat_dir} ...")
    by_spk = {s: load_speakers(args.feat_dir, [s]) for s in ALL_SPEAKERS}
    print({s: len(v) for s, v in by_spk.items()})

    for name in args.variants:
        accs, f1s = [], []
        for fold in args.folds:
            tr_spk, te_spk = fold_split(fold)
            train = [r for s in tr_spk for r in by_spk[s]]
            test = [r for s in te_spk for r in by_spk[s]]
            assert not set(tr_spk) & set(te_spk)
            if name == "frameattn" and "frames" not in train[0]:
                sys.exit("frameattn needs features extracted with --keep_frames")
            print(f"\n=== {name} | fold {fold} | train {len(train)} utts / test {len(test)} "
                  f"({','.join(te_spk)})")
            torch.manual_seed(args.seed)
            model = build_model(name, in_dim=args.in_dim)
            nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  trainable params: {nparam:,}")
            rows, m, hist = train_fold(model, train, test, args.device, args.epochs, args.lr,
                                       args.batch_size, args.seed,
                                       log_test_curve=args.log_test_curve)
            print(f"  >>> acc={m['acc']:.2f}  macro-F1={m['macro_f1']:.2f}")
            d = os.path.join(args.out, name, f"fold{fold}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "preds.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["utt", "spk", "y", "pred"])
                w.writeheader()
                w.writerows(rows)
            with open(os.path.join(d, "metrics.json"), "w") as f:
                json.dump({"metrics": m, "history": hist, "params": nparam,
                           "args": vars(args)}, f, indent=2)
            accs.append(m["acc"])
            f1s.append(m["macro_f1"])
        print(f"\n##### {name}: acc {np.mean(accs):.2f} ± {np.std(accs):.2f} | "
              f"macro-F1 {np.mean(f1s):.2f} ± {np.std(f1s):.2f}  (population std over folds)")


if __name__ == "__main__":
    main()

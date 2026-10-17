"""Summarise runs/<exp>/<variant>/fold*/preds.csv.

  python scripts/report.py runs/whisper                       # table of all variants
  python scripts/report.py runs/whisper --per_speaker paid    # accuracy for each of 24 speakers
  python scripts/report.py runs/whisper --compare paid utt    # paired bootstrap over SPEAKERS

Why bootstrap over speakers: the effective sample size is 24 speakers, not ~27k utterances
(utterances of one speaker are strongly correlated). Resampling speakers with replacement
gives an honest uncertainty estimate for the difference between two systems.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paid.l2arctic import L1S, SPK_TO_L1  # noqa: E402


def load_preds(run_dir, variant):
    files = sorted(glob.glob(os.path.join(run_dir, variant, "fold*", "preds.csv")))
    if not files:
        raise SystemExit(f"no predictions for '{variant}' under {run_dir}")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["fold"] = int(os.path.basename(os.path.dirname(f))[4:])
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def fold_metrics(df):
    rows = []
    for k, g in df.groupby("fold"):
        rows.append({"fold": k, "acc": 100 * (g.y == g.pred).mean(),
                     "macro_f1": 100 * f1_score(g.y, g.pred, average="macro",
                                                labels=list(range(len(L1S))), zero_division=0)})
    return pd.DataFrame(rows)


def per_speaker_acc(df):
    s = df.assign(ok=(df.y == df.pred)).groupby("spk").ok.mean() * 100
    return s


def paired_bootstrap(df_a, df_b, n_boot=10000, seed=0):
    """Difference in utterance-level accuracy (a - b), resampling speakers."""
    a, b = per_speaker_acc(df_a), per_speaker_acc(df_b)
    spk = sorted(set(a.index) & set(b.index))
    d = (a[spk] - b[spk]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return d.mean(), np.percentile(boots, [2.5, 97.5]), (boots <= 0).mean(), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--per_speaker", metavar="VARIANT")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.per_speaker:
        s = per_speaker_acc(load_preds(args.run_dir, args.per_speaker))
        print(pd.DataFrame({"L1": [SPK_TO_L1[x] for x in s.index], "acc": s.round(1)})
              .sort_values(["L1", "acc"]).to_string())
        return
    if args.compare:
        a, b = args.compare
        mean_d, ci, p, d = paired_bootstrap(load_preds(args.run_dir, a),
                                            load_preds(args.run_dir, b))
        print(f"{a} - {b}: mean per-speaker acc diff = {mean_d:+.2f} pts, "
              f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], P(diff<=0) = {p:.3f}")
        print(f"speakers where {a} better / worse: {(d > 0).sum()} / {(d < 0).sum()} of {len(d)}")
        return

    out = []
    for v in sorted(os.listdir(args.run_dir)):
        if not os.path.isdir(os.path.join(args.run_dir, v)):
            continue
        m = fold_metrics(load_preds(args.run_dir, v))
        out.append({"variant": v, "folds": len(m),
                    "acc": f"{m.acc.mean():.2f} ± {m.acc.std(ddof=0):.2f}",
                    "macro_f1": f"{m.macro_f1.mean():.2f} ± {m.macro_f1.std(ddof=0):.2f}",
                    "per-fold acc": " ".join(f"{x:.1f}" for x in m.acc)})
    print(pd.DataFrame(out).to_string(index=False))
    print("(std = population std over folds, ddof=0)")


if __name__ == "__main__":
    main()

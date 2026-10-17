
import argparse
from pathlib import Path

import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

from paid.l2arctic import L1S, SPK_TO_L1

L1_ID_TO_NAME = dict(enumerate(L1S))
# Collect predictions from all 4 folds.
def load_variant(run, variant):
    dfs = []
    for fold in range(4):
        path = run / variant / f"fold{fold}" / "preds.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def add_l1_columns(df):
    # y/pred are numeric class IDs.
    df["true_l1"] = df["y"].map(L1_ID_TO_NAME)
    df["pred_l1"] = df["pred"].map(L1_ID_TO_NAME)
    assert df["true_l1"].notna().all() and df["pred_l1"].notna().all(), \
        "y/pred contain IDs outside 0..5; label order may differ from training"
    unknown = set(df["spk"]) - set(SPK_TO_L1)
    assert not unknown, f"unknown speakers in preds.csv: {unknown}"
    return df
def per_speaker_acc(df):
    return (df["true_l1"] == df["pred_l1"]).groupby(df["spk"]).mean().mul(100)


def load_optional(run, variant):
    """Đọc variant nếu đủ preds.csv của 4 fold, ngược lại trả về None."""
    if not all((run / variant / f"fold{k}" / "preds.csv").exists() for k in range(4)):
        print(f"[bỏ qua] {variant}: thiếu preds.csv ở ít nhất một fold")
        return None
    return add_l1_columns(load_variant(run, variant))


def print_fold_summary(variants):
    print("\n" + "=" * 70)
    print("0. ACCURACY THEO FOLD (mean ± std, ddof=0)")
    print("=" * 70)
    rows = {}
    for name, df in variants:
        correct = df["true_l1"] == df["pred_l1"]
        fa = correct.groupby(df["fold"]).mean().mul(100)
        row = {f"fold{k}": v for k, v in fa.items()}
        row["mean"] = fa.mean()
        row["std"] = fa.std(ddof=0)
        row["pooled"] = correct.mean() * 100
        rows[name] = row
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n'mean' là trung bình 4 fold (số để so với bài báo); "
          "'pooled' là accuracy gộp mọi utterance.")


def main(run):
    paid = add_l1_columns(load_variant(run, "paid"))
    utt = add_l1_columns(load_variant(run, "utt"))
    nopid = add_l1_columns(load_variant(run, "paid_nopid"))

    labels = list(L1_ID_TO_NAME.values())
    sup = load_optional(run, "speechutt_proj")

    all_variants = [("PAID", paid), ("PAID-noPID", nopid), ("UTT", utt)]
    if sup is not None:
        all_variants.append(("SpeechUtt-proj", sup))

    print_fold_summary(all_variants)
    print("\n" + "=" * 70)
    print("1. CLASSIFICATION REPORT")
    print("=" * 70)

    for name, df in [
        ("PAID", paid),
        ("PAID-noPID", nopid),
        ("UTT", utt),
    ]:
        print(f"\n--- {name} ---")
        print(
            classification_report(
                df["true_l1"],
                df["pred_l1"],
                labels=labels,
                target_names=labels,
                digits=4,
                zero_division=0,
            )
        )

    print("\n" + "=" * 70)
    print("2. CONFUSION MATRICES")
    print("=" * 70)

    for name, df in [
        ("PAID", paid),
        ("PAID-noPID", nopid),
        ("UTT", utt),
    ]:
        cm = confusion_matrix(
            df["true_l1"],
            df["pred_l1"],
            labels=labels,
        )

        print(f"\n--- {name} ---")
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        print(cm_df)

    print("\n" + "=" * 70)
    print("3. PER-SPEAKER ACCURACY")
    print("=" * 70)

    for name, df in [
        ("PAID", paid),
        ("PAID-noPID", nopid),
        ("UTT", utt),
    ]:
        spk_acc = (
            df.assign(correct=df["true_l1"] == df["pred_l1"])
              .groupby("spk")["correct"]
              .mean()
              .mul(100)
              .sort_values()
        )

        print(f"\n--- {name} ---")
        for spk, acc in spk_acc.items():
            print(f"{spk:5s} {SPK_TO_L1[spk]:10s} {acc:6.2f}%")

    print("\n" + "=" * 70)
    print("4. SPEAKER-BY-SPEAKER: PAID so với các variant khác")
    print("=" * 70)
    print("Chỉ mang tính mô tả; kết luận thống kê dùng scripts/report.py --compare.")

    paid_spk = per_speaker_acc(paid)
    for other_name, other_df in all_variants[1:]:
        col = f"PAID_minus_{other_name}"
        comparison = pd.DataFrame({
            "L1": pd.Series(SPK_TO_L1),
            "PAID": paid_spk,
            other_name: per_speaker_acc(other_df),
        })
        comparison[col] = comparison["PAID"] - comparison[other_name]
        print(f"\n--- PAID vs {other_name} ---")
        print(comparison.sort_values(col).to_string(float_format=lambda x: f"{x:.2f}"))



    print("\n" + "=" * 70)
    print("5. FOLD 3 DETAIL")
    print("=" * 70)

    for name, df in [
        ("PAID", paid),
        ("UTT", utt),
    ]:
        fold3 = df[df["fold"] == 3].copy()
        print(f"\n--- {name} / Fold 3 ---")

        fold3_spk = (
            fold3.assign(correct=fold3["true_l1"] == fold3["pred_l1"])
                 .groupby("spk")["correct"]
                 .mean()
                 .mul(100)
                 .sort_values()
        )

        for spk, acc in fold3_spk.items():
            print(f"{spk:5s} {SPK_TO_L1[spk]:10s} {acc:6.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True,
                        help="thư mục run, ví dụ runs/whisper")
    args = parser.parse_args()
    main(args.run)
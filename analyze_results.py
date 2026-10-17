
import argparse
from pathlib import Path

import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

SPK_TO_L1 = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Chinese", "LXC": "Chinese", "NCC": "Chinese", "TXHC": "Chinese",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}

L1_ID_TO_NAME = {
    0: "Arabic",
    1: "Chinese",
    2: "Hindi",
    3: "Korean",
    4: "Spanish",
    5: "Vietnamese",
}

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
    return df


def main(run):
    paid = add_l1_columns(load_variant(run, "paid"))
    utt = add_l1_columns(load_variant(run, "utt"))
    nopid = add_l1_columns(load_variant(run, "paid_nopid"))

    labels = list(L1_ID_TO_NAME.values())

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
    print("4. SPEAKER-BY-SPEAKER PAID vs UTT")
    print("=" * 70)

    paid_spk = (
        paid.assign(correct=paid["true_l1"] == paid["pred_l1"])
             .groupby("spk")["correct"]
             .mean()
             .mul(100)
    )

    utt_spk = (
        utt.assign(correct=utt["true_l1"] == utt["pred_l1"])
            .groupby("spk")["correct"]
            .mean()
            .mul(100)
    )

    comparison = pd.DataFrame({
        "L1": pd.Series(SPK_TO_L1),
        "PAID": paid_spk,
        "UTT": utt_spk,
    })

    comparison["PAID_minus_UTT"] = comparison["PAID"] - comparison["UTT"]
    comparison = comparison.sort_values("PAID_minus_UTT")

    print(comparison.to_string(float_format=lambda x: f"{x:.2f}"))

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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paid.l2arctic import ALL_SPEAKERS, L1S, N_FOLDS, SPK_TO_L1, fold_split  # noqa: E402
from paid.phones import ARPABET_39, PHONE2ID, normalize_phone  # noqa: E402
from paid.textgrid import read_alignment  # noqa: E402

TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = "this"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 6
        intervals [1]:
            xmin = 0
            xmax = 0.25
            text = ""
        intervals [2]:
            xmin = 0.25
            xmax = 0.5
            text = "sil"
        intervals [3]:
            xmin = 0.5
            xmax = 0.6
            text = "DH"
        intervals [4]:
            xmin = 0.6
            xmax = 0.8
            text = "IH1"
        intervals [5]:
            xmin = 0.8
            xmax = 0.9
            text = "spn"
        intervals [6]:
            xmin = 0.9
            xmax = 1.0
            text = "S"
'''


def test_inventory():
    assert len(ARPABET_39) == 39
    assert normalize_phone("AH0") == "AH" and normalize_phone("ih1") == "IH"
    assert normalize_phone("sil") is None and normalize_phone("") is None
    assert normalize_phone("spn") is None and normalize_phone("XX") is None


def test_textgrid_filters_silence_and_stress(tmp_path):
    p = tmp_path / "a.TextGrid"
    p.write_text(TG)
    ali = read_alignment(str(p))
    assert ali == [(PHONE2ID["DH"], 0.5, 0.6), (PHONE2ID["IH"], 0.6, 0.8),
                   (PHONE2ID["S"], 0.9, 1.0)]


def test_folds_are_speaker_disjoint_and_balanced():
    assert len(ALL_SPEAKERS) == 24
    seen_test = []
    for k in range(N_FOLDS):
        tr, te = fold_split(k)
        assert not set(tr) & set(te)
        assert len(te) == 6 and sorted(SPK_TO_L1[s] for s in te) == L1S   # 1 per L1
        assert len(tr) == 18 and set(tr) | set(te) == set(ALL_SPEAKERS)
        seen_test += te
    assert sorted(seen_test) == sorted(ALL_SPEAKERS)    # every speaker tested exactly once

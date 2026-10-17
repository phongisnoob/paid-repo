"""Phone inventory: stress-stripped ARPAbet, P = 39 (no silence / padding symbols)."""
import re

ARPABET_39 = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
assert len(ARPABET_39) == 39 and len(set(ARPABET_39)) == 39
PHONE2ID = {p: i for i, p in enumerate(ARPABET_39)}
NON_SPEECH = {"", "sil", "sp", "spn", "<eps>", "nsn", "noise"}


def normalize_phone(label: str):
    """MFA label -> ARPAbet-39 symbol, or None if silence/pause/unknown.

    'AH0' -> 'AH', 'sil' -> None, 'spn' (spoken noise / OOV) -> None.
    """
    lab = label.strip()
    if lab.lower() in NON_SPEECH:
        return None
    lab = re.sub(r"_[BEIS]$", "", lab)       # position suffixes of old MFA models
    lab = re.sub(r"\d+$", "", lab).upper()   # stress digits
    return lab if lab in PHONE2ID else None

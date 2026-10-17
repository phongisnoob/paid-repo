"""Minimal parser for Praat long-format TextGrids as written by MFA, + phone filtering."""
import re

from .phones import PHONE2ID, normalize_phone

_INTERVAL = re.compile(
    r'xmin\s*=\s*([-\d.eE+]+)\s*\n\s*xmax\s*=\s*([-\d.eE+]+)\s*\n\s*text\s*=\s*"([^"]*)"')


def read_tier(path: str, tier_suffix: str = "phones"):
    """Return [(xmin, xmax, text)] of the first tier whose name ends with `tier_suffix`."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for item in re.split(r"\n\s*item \[\d+\]:", text)[1:]:
        m = re.search(r'name\s*=\s*"([^"]*)"', item)
        if m and m.group(1).endswith(tier_suffix):
            return [(float(a), float(b), t) for a, b, t in _INTERVAL.findall(item)]
    raise ValueError(f"no tier ending with '{tier_suffix}' in {path} "
                     "(is it a long-format TextGrid from MFA?)")


def read_alignment(path: str):
    """TextGrid -> [(phone_id, start, end)] keeping only valid ARPAbet-39 intervals."""
    out = []
    for a, b, t in read_tier(path, "phones"):
        p = normalize_phone(t)
        if p is not None and b > a:
            out.append((PHONE2ID[p], a, b))
    return out

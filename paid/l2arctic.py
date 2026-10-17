"""L2-ARCTIC metadata and the 4-fold speaker-disjoint protocol.

The paper only says: 'each fold holds out one speaker from each L1 group'.
It does NOT say which speaker goes to which fold, so fold k = k-th speaker of each
L1 in the order below. Per-fold numbers therefore cannot match the paper exactly.
"""
SPEAKERS_BY_L1 = {
    "Arabic":     ["ABA", "SKA", "YBAA", "ZHAA"],
    "Chinese":    ["BWC", "LXC", "NCC", "TXHC"],
    "Hindi":      ["ASI", "RRBI", "SVBI", "TNI"],
    "Korean":     ["HJK", "HKK", "YDCK", "YKWK"],
    "Spanish":    ["EBVS", "ERMS", "MBMPS", "NJS"],
    "Vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}
L1S = sorted(SPEAKERS_BY_L1)                       # fixed label order
L1_TO_ID = {l1: i for i, l1 in enumerate(L1S)}
SPK_TO_L1 = {s: l1 for l1, spks in SPEAKERS_BY_L1.items() for s in spks}
ALL_SPEAKERS = [s for l1 in L1S for s in SPEAKERS_BY_L1[l1]]
N_FOLDS = 4


def spk_label(spk: str) -> int:
    return L1_TO_ID[SPK_TO_L1[spk]]


def fold_split(fold: int):
    """Return (train_speakers, test_speakers) for the given fold."""
    assert 0 <= fold < N_FOLDS
    test = [SPEAKERS_BY_L1[l1][fold] for l1 in L1S]
    train = [s for s in ALL_SPEAKERS if s not in test]
    return train, test

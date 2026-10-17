"""Loading cached features and collating batches.

Feature cache layout (written by scripts/extract_features.py):
    <feat_dir>/<SPEAKER>.pt   ->  list of records (dict) from paid.pooling.build_record
                                  each extended with 'utt' (str) and 'spk' (str).
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset

from .l2arctic import spk_label


def load_speaker(feat_dir: str, spk: str):
    path = os.path.join(feat_dir, f"{spk}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - run scripts/extract_features.py first")
    recs = torch.load(path, map_location="cpu", weights_only=False)
    y = spk_label(spk)
    for r in recs:
        r["spk"], r["y"] = spk, y
    return recs


def load_speakers(feat_dir: str, speakers):
    out = []
    for s in speakers:
        out.extend(load_speaker(feat_dir, s))
    return out


class RecordDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


def _pad(seqs, dtype):
    """Right-pad a list of [n_i, ...] tensors -> ([B, max_n, ...], bool mask [B, max_n])."""
    n = max(s.shape[0] for s in seqs)
    shape = (len(seqs), n) + tuple(seqs[0].shape[1:])
    out = torch.zeros(shape, dtype=dtype)
    mask = torch.zeros(len(seqs), n, dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s.to(dtype)
        mask[i, : s.shape[0]] = True
    return out, mask


def collate(batch):
    tokens, mask = _pad([b["tokens"] for b in batch], torch.float32)
    phones, _ = _pad([b["phones"] for b in batch], torch.long)
    out = {
        "tokens": tokens, "phones": phones, "mask": mask,
        "utt_mean": torch.stack([b["utt_mean"] for b in batch]).float(),
        "speech_mean": torch.stack([b["speech_mean"] for b in batch]).float(),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
        "spk": [b["spk"] for b in batch],
        "utt": [b.get("utt", "") for b in batch],
    }
    if "frames" in batch[0]:
        out["frames"], out["fmask"] = _pad([b["frames"] for b in batch], torch.float32)
    return out


def make_loader(records, batch_size, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(RecordDataset(records), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=g, num_workers=0)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

"""Models. Every model takes the collated `batch` dict (see paid/data.py) so that
training / evaluation code is model-agnostic.

PAIDModel(fusion="concat_linear") is the architecture exactly as described in the paper:
    r_i   = Dropout(LayerNorm(Linear(z_i)))        # 768 -> 192
    r~_i  = [r_i ; Emb[p_i]]                        # 192 + 64 = 256
    u     = masked_mean(r~_i)
    logits= Linear(u)
NOTE: because masked-mean and Linear commute, this is mathematically
    logits = W_r mean_i(r_i) + W_e mean_i(Emb[p_i]) + b
i.e. NO interaction between the acoustics and the phoneme identity. `mlp` and
`transformer` fusions add that interaction before pooling.
"""
import torch
import torch.nn as nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [B,M,D], mask [B,M] bool -> [B,D]; padded positions ignored."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class ProjBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


def shuffle_ids_within_utt(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Randomly permute phone ids among the VALID positions of each utterance.
    Valid positions are always a prefix (right padding), so permuting the first n_b entries
    is enough. Keeps the multiset of phones (bag-of-phonemes) but destroys the alignment
    between a token's acoustics and its phone id."""
    scores = torch.rand(ids.shape, device=ids.device)
    scores = scores.masked_fill(~mask, 2.0)       # padding sorts last
    perm = scores.argsort(dim=1)
    return ids.gather(1, perm)


class PAIDModel(nn.Module):
    def __init__(self, in_dim=768, proj_dim=192, pid_dim=64, n_phones=39, n_classes=6,
                 dropout=0.1, use_phone_id=True, shuffle_phone_id=False,
                 fusion="concat_linear", hidden=256, n_layers=2, n_heads=4):
        super().__init__()
        assert fusion in ("concat_linear", "mlp", "transformer")
        self.fusion, self.use_phone_id, self.shuffle_phone_id = fusion, use_phone_id, shuffle_phone_id
        self.proj = ProjBlock(in_dim, proj_dim, dropout)
        self.emb = nn.Embedding(n_phones, pid_dim) if use_phone_id else None
        d = proj_dim + (pid_dim if use_phone_id else 0)
        if fusion == "concat_linear":
            self.fuse, d_out = nn.Identity(), d
        elif fusion == "mlp":
            self.fuse = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, hidden), nn.GELU())
            d_out = hidden
        else:
            layer = nn.TransformerEncoderLayer(hidden, n_heads, dim_feedforward=2 * hidden,
                                               dropout=dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.fuse = nn.ModuleDict({
                "inp": nn.Linear(d, hidden),
                "enc": nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False),
            })
            d_out = hidden
        self.cls = nn.Linear(d_out, n_classes)

    def forward(self, batch):
        x, ids, mask = batch["tokens"], batch["phones"], batch["mask"]
        r = self.proj(x)
        if self.use_phone_id:
            if self.shuffle_phone_id:
                ids = shuffle_ids_within_utt(ids, mask)
            r = torch.cat([r, self.emb(ids)], dim=-1)
        if self.fusion == "transformer":
            r = self.fuse["enc"](self.fuse["inp"](r), src_key_padding_mask=~mask)
        else:
            r = self.fuse(r)
        return self.cls(masked_mean(r, mask))


class UttModel(nn.Module):
    """Utterance-level baselines on a single pooled vector.
    key='utt_mean'    : mean over ALL real frames (what 'Whisper-Utt' most likely is)
    key='speech_mean' : mean over frames inside phoneme spans only (control: same silence
                        removal as PAID, but no span grouping / phone id)
    use_proj=False -> Linear only; use_proj=True -> same ProjBlock as PAID, then Linear."""

    def __init__(self, key="utt_mean", in_dim=768, proj_dim=192, n_classes=6, dropout=0.1,
                 use_proj=False):
        super().__init__()
        self.key = key
        if use_proj:
            self.net = nn.Sequential(ProjBlock(in_dim, proj_dim, dropout),
                                     nn.Linear(proj_dim, n_classes))
        else:
            self.net = nn.Linear(in_dim, n_classes)

    def forward(self, batch):
        return self.net(batch[self.key])


class FrameAttnModel(nn.Module):
    """Frame-level attention pooling baseline (needs --keep_frames features)."""

    def __init__(self, in_dim=768, proj_dim=192, n_classes=6, dropout=0.1):
        super().__init__()
        self.proj = ProjBlock(in_dim, proj_dim, dropout)
        self.att = nn.Linear(proj_dim, 1)
        self.cls = nn.Linear(proj_dim, n_classes)

    def forward(self, batch):
        r = self.proj(batch["frames"])
        a = self.att(r).squeeze(-1).masked_fill(~batch["fmask"], float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return self.cls((w * r).sum(1))


# ----------------------------------------------------------------------------- variants
VARIANTS = {
    # --- faithful reproduction -------------------------------------------------------
    "paid":              lambda **k: PAIDModel(**k),
    # --- ablations from the paper ----------------------------------------------------
    "paid_nopid":        lambda **k: PAIDModel(use_phone_id=False, **k),
    "utt":               lambda **k: UttModel("utt_mean", use_proj=False, **k),
    "frameattn":         lambda **k: FrameAttnModel(**k),
    # --- additional controls that the paper is missing -------------------------------
    "utt_proj":          lambda **k: UttModel("utt_mean", use_proj=True, **k),
    "speechutt":         lambda **k: UttModel("speech_mean", use_proj=False, **k),
    "speechutt_proj":    lambda **k: UttModel("speech_mean", use_proj=True, **k),
    # shuffle control: for the faithful model this is PROVABLY a no-op (bag-of-phonemes, see
    # tests/test_models.py) - it is kept as a sanity check. It only bites for models that
    # combine acoustics and phone id per token (mlp / transformer).
    "paid_shuffle_pid":  lambda **k: PAIDModel(shuffle_phone_id=True, **k),
    "paid_mlp":          lambda **k: PAIDModel(fusion="mlp", **k),
    "paid_tf":           lambda **k: PAIDModel(fusion="transformer", **k),
    "paid_mlp_nopid":    lambda **k: PAIDModel(fusion="mlp", use_phone_id=False, **k),
    "paid_tf_nopid":     lambda **k: PAIDModel(fusion="transformer", use_phone_id=False, **k),
    "paid_mlp_shuffle_pid": lambda **k: PAIDModel(fusion="mlp", shuffle_phone_id=True, **k),
    "paid_tf_shuffle_pid":  lambda **k: PAIDModel(fusion="transformer", shuffle_phone_id=True, **k),
}


def build_model(name: str, **kwargs):
    return VARIANTS[name](**kwargs)

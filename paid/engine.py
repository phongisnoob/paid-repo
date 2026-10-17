"""Training and evaluation of one fold. Mirrors the paper's recipe by default:
Adam, lr 2e-4, batch 16, 10 epochs, dropout 0.1, seed 1337, NO validation / early stopping
(the paper has no validation set, so the number of epochs is fixed)."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .data import make_loader, to_device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, records, device, batch_size=64):
    model.eval()
    rows = []
    for batch in make_loader(records, batch_size, shuffle=False):
        b = to_device(batch, device)
        pred = model(b).argmax(-1).cpu().tolist()
        for u, s, y, p in zip(batch["utt"], batch["spk"], batch["y"].tolist(), pred):
            rows.append({"utt": u, "spk": s, "y": y, "pred": p})
    return rows


def metrics_from_rows(rows, n_classes=6):
    y = [r["y"] for r in rows]
    p = [r["pred"] for r in rows]
    return {
        "acc": 100 * accuracy_score(y, p),
        "macro_f1": 100 * f1_score(y, p, average="macro", labels=list(range(n_classes)),
                                   zero_division=0),
        "n": len(rows),
    }


def train_fold(model, train_recs, test_recs, device, epochs=10, lr=2e-4, batch_size=16,
               seed=1337, log_fn=print, log_test_curve=False):
    """Returns (pred_rows, final_metrics, history). Test data is touched only after the last
    epoch unless log_test_curve=True (diagnostic only - never use it to pick an epoch)."""
    set_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in make_loader(train_recs, batch_size, shuffle=True, seed=seed + ep):
            b = to_device(batch, device)
            loss = F.cross_entropy(model(b), b["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b["y"])
            n += len(b["y"])
        h = {"epoch": ep, "train_loss": tot / n}
        if log_test_curve:
            h.update({f"test_{k}": v for k, v in
                      metrics_from_rows(predict(model, test_recs, device)).items()})
        history.append(h)
        log_fn("  " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in h.items()))
    rows = predict(model, test_recs, device)
    return rows, metrics_from_rows(rows), history

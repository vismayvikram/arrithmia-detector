import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
VAL_RECS = [106, 203, 215, 223]

SEEDS = [42, 123, 999]
MAX_EPOCHS, PATIENCE, BATCH, LR = 15, 5, 256, 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

data = np.load("data_cache.npz")
X, rr, y, rec = data["X"].astype(np.float32), data["rr"].astype(np.float32), data["y"].astype(int), data["rec"].astype(int)
print(f"Total beats: {len(y)} | Class counts: {np.bincount(y, minlength=5)}")


class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.rr_fc = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 16, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 5))

    def forward(self, x, r):
        x = self.conv(x.unsqueeze(1)).mean(dim=2)
        return self.head(torch.cat([x, self.rr_fc(r)], dim=1))

def predict(model, Xs, rs):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), 2048):
            xb, rb = torch.tensor(Xs[i:i+2048]).to(DEVICE), torch.tensor(rs[i:i+2048]).to(DEVICE)
            preds.append(model(xb, rb).argmax(1).cpu().numpy())
    return np.concatenate(preds)

def macro_f1_4(yt, yp):
    return f1_score(yt, yp, labels=[0, 1, 2, 3], average="macro", zero_division=0)

def train_and_test(tr, va, te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PlainCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    counts = np.maximum(np.bincount(y[tr], minlength=5), 1)
    w = torch.tensor(np.minimum(np.sqrt(len(tr) / counts), 10), dtype=torch.float32)
    w = (w / w.mean()).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    Xtr, rtr, ytr = torch.tensor(X[tr]), torch.tensor(rr[tr]), torch.tensor(y[tr])
    best_f1, best_state, bad = -1, None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(tr))
        for i in range(0, len(tr), BATCH):
            idx = order[i:i+BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(DEVICE), rtr[idx].to(DEVICE)), ytr[idx].to(DEVICE))
            loss.backward()
            opt.step()

        vf1 = macro_f1_4(y[va], predict(model, X[va], rr[va]))
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break

    model.load_state_dict(best_state)
    tp = predict(model, X[te], rr[te])
    return {
        "accuracy": accuracy_score(y[te], tp),
        "macro_f1_5class": f1_score(y[te], tp, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0),
        "macro_f1_4class": macro_f1_4(y[te], tp),
    }

ip_tr = np.where(np.isin(rec, [r for r in DS1 if r not in VAL_RECS]))[0]
ip_va = np.where(np.isin(rec, VAL_RECS))[0]
ip_te = np.where(np.isin(rec, DS2))[0]
print(f"Inter-patient sizes -> Train: {len(ip_tr)} | Val: {len(ip_va)} | Test: {len(ip_te)}")

def make_naive_split(seed):
    pool = np.where(np.isin(rec, DS1 + DS2))[0]
    rest, te = train_test_split(pool, test_size=len(ip_te), stratify=y[pool], random_state=seed)
    tr, va = train_test_split(rest, test_size=len(ip_va), stratify=y[rest], random_state=seed)
    return tr, va, te

rows = []
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    for name, (tr, va, te) in [
        ("naive random (leaky)", make_naive_split(seed)), 
        ("inter-patient (DS1->DS2)", (ip_tr, ip_va, ip_te))
    ]:
        res = train_and_test(tr, va, te, seed)
        rows.append({"split": name, "seed": seed, **res})
        print(f"{name:<26} | Acc: {res['accuracy']:.4f} | F1-4c: {res['macro_f1_4class']:.4f}")

df = pd.DataFrame(rows)
summary = df.groupby("split")[["accuracy", "macro_f1_5class", "macro_f1_4class"]].agg(["mean", "std"])

print("\n" + "="*55 + "\n SUMMARY (Mean ± Std over Seeds)\n" + "="*55)
print(summary.round(4))
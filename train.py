#Training Pipeline for Arrhythmia Detection Models


import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data import build_dataset, get_splits, COST_MATRIX, CLASS_NAMES
from models import PlainCNN, ECGAttentionCNN, WideCNN, count_parameters

SEEDS = [42, 123, 999]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
MAX_EPOCHS = 25
PATIENCE = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    v_rec = recall_score(y_true == 2, y_pred == 2, zero_division=0)
    v_prec = precision_score(y_true == 2, y_pred == 2, zero_division=0)

    total_cost = sum(COST_MATRIX[t, p] for t, p in zip(y_true, y_pred))
    return {
        'macro_f1': macro_f1,
        'v_recall': v_rec,
        'v_precision': v_prec,
        'avg_cost': total_cost / len(y_true)
    }


def evaluate_torch_model(model, loader, device=DEVICE):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for bx, brr, by in loader:
            bx, brr = bx.to(device), brr.to(device)
            out = model(bx, brr)
            logits = out[0] if isinstance(out, tuple) else out
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_trues.extend(by.numpy())

    return np.array(all_trues), np.array(all_preds)


def train_single_neural_model(model_cls, X_train, rr_train, y_train,
                               X_val, rr_val, y_val,
                               class_weights, lr=1e-3, dropout=0.3,
                               save_path=None, use_weights=True):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(rr_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(rr_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = model_cls(num_classes=5, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_f1, best_weights, epochs_no_improve = -1.0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for bx, brr, by in train_loader:
            bx, brr, by = bx.to(DEVICE), brr.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx, brr)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Validation check
        y_val_true, y_val_pred = evaluate_torch_model(model, val_loader, device=DEVICE)
        val_f1 = compute_metrics(y_val_true, y_val_pred)['macro_f1']

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return model, best_val_f1


def run_training_pipeline():
    os.makedirs('results', exist_ok=True)
    print(" MIT-BIH ARRHYTHMIA DETECTION TRAINING PIPELINE ")

    # 1. Load data & splits
    X, rr, y, rec = build_dataset()
    train_mask, val_mask, test_mask = get_splits(rec, y)

    X_train, rr_train, y_train = X[train_mask], rr[train_mask], y[train_mask]
    X_val, rr_val, y_val = X[val_mask], rr[val_mask], y[val_mask]
    X_test, rr_test, y_test = X[test_mask], rr[test_mask], y[test_mask]

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
            torch.tensor(rr_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # 2. Compute class weights
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = len(y_train) / (5.0 * np.maximum(class_counts, 1))
    print(f"Class counts in DS1 Train: {class_counts}")
    print(f"Computed loss weights:     {np.round(class_weights, 3)}")

    all_results = []
    best_lr, best_dropout = 1e-3, 0.3

    #Random Forest on RR features only
    print("\n--- Training Baseline 1: Random Forest (RR features only) ---")
    for seed in SEEDS:
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
        rf.fit(rr_train, y_train)
        m = compute_metrics(y_test, rf.predict(rr_test))
        all_results.append({'model': 'RandomForest (RR only)', 'seed': seed, **m})
        print(f"  Seed {seed} - Macro F1: {m['macro_f1']:.4f}, V Recall: {m['v_recall']:.4f}, Avg Cost: {m['avg_cost']:.3f}")

    neural_configs = [
        ("Plain CNN", PlainCNN, "results/plain_cnn_seed{}.pt", True),
        ("ECGAttentionCNN", ECGAttentionCNN, "results/ecg_attention_seed{}.pt", True),
        ("WideCNN (matched param)", WideCNN, "results/wide_cnn_seed{}.pt", True),
        ("ECGAttention (Unweighted Loss)", ECGAttentionCNN, "results/ecg_attention_unweighted.pt", False)
    ]

    for model_name, model_cls, path_template, use_w in neural_configs:
        print(f"\n--- Training: {model_name} ---")
        
        # If unweighted loss, run only for seed 42
        target_seeds = [42] if "Unweighted" in model_name else SEEDS

        for seed in target_seeds:
            set_seed(seed)
            save_path = path_template.format(seed) if "{}" in path_template else path_template
            
            model, best_val_f1 = train_single_neural_model(
                model_cls, X_train, rr_train, y_train,
                X_val, rr_val, y_val, class_weights if use_w else None,
                lr=best_lr, dropout=best_dropout, save_path=save_path, use_weights=use_w
            )
            
            _, preds = evaluate_torch_model(model, test_loader, device=DEVICE)
            m = compute_metrics(y_test, preds)
            all_results.append({'model': model_name, 'seed': seed, **m})
            print(f"  Seed {seed} (val F1 {best_val_f1:.4f}) -> Test Macro F1: {m['macro_f1']:.4f}, V Recall: {m['v_recall']:.4f}, Avg Cost: {m['avg_cost']:.3f}")

    # Save and summarize results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('results/training_results.csv', index=False)
    print("\nTraining complete! Raw seed metrics saved to results/training_results.csv")

    summary = df_results[df_results['seed'].isin(SEEDS)].groupby('model').agg({
        'macro_f1': ['mean', 'std'],
        'v_recall': ['mean', 'std'],
        'v_precision': ['mean', 'std'],
        'avg_cost': ['mean', 'std']
    })
    print("\n" + "=" * 60)
    print(" SUMMARY (Mean ± Std over 3 seeds) ")
    print("=" * 60)
    print(summary.round(4))


if __name__ == '__main__':
    run_training_pipeline()
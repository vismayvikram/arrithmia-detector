

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, recall_score, precision_score
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data import build_dataset, get_splits, COST_MATRIX, CLASS_NAMES
from models import PlainCNN, ECGAttentionCNN, WideCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 999]


def load_model(model_cls, checkpoint_path):
    model = model_cls(num_classes=5)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def get_model_predictions_and_probs(model, X, rr):
    """Returns predictions and softmax probability distributions."""
    model.eval()
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32).unsqueeze(1),
        torch.tensor(rr, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_probs = []

    with torch.no_grad():
        for bx, brr in loader:
            bx, brr = bx.to(DEVICE), brr.to(DEVICE)
            out = model(bx, brr)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def plot_confusion_matrix(y_true, y_pred, save_path='results/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, cmap='Blues', interpolation='nearest')
    plt.title('Normalized Confusion Matrix (Best ECGAttentionCNN on DS2)')
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES)
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black",
                     fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def evaluate_pr_curve(val_probs, y_val, test_probs, y_test, save_path='results/pr_curve_v_class.png'):
    # V is index 2
    v_val_true = (y_val == 2).astype(int)
    v_val_scores = val_probs[:, 2]

    prec_val, rec_val, thresholds_val = precision_recall_curve(v_val_true, v_val_scores)

    # Pick threshold on val that gives at least 90% recall
    target_rec = 0.90
    valid_idx = np.where(rec_val >= target_rec)[0]
    if len(valid_idx) > 0:
        chosen_thresh = thresholds_val[valid_idx[-1]]
    else:
        chosen_thresh = 0.5

    # Evaluate on test set
    v_test_true = (y_test == 2).astype(int)
    v_test_scores = test_probs[:, 2]
    prec_test, rec_test, _ = precision_recall_curve(v_test_true, v_test_scores)

    # Test performance at default 0.5 vs chosen threshold
    pred_default = (v_test_scores >= 0.5).astype(int)
    pred_tuned = (v_test_scores >= chosen_thresh).astype(int)

    def get_metrics_binary(t, p):
        return precision_score(t, p, zero_division=0), recall_score(t, p, zero_division=0)

    p_def, r_def = get_metrics_binary(v_test_true, pred_default)
    p_tune, r_tune = get_metrics_binary(v_test_true, pred_tuned)

    plt.figure(figsize=(7, 5))
    plt.plot(rec_test, prec_test, label='DS2 Test PR Curve (ECGAttentionCNN)', color='darkblue', lw=2)
    plt.scatter([r_def], [p_def], color='red', s=70, zorder=5, label=f'Default (thresh=0.5): Rec={r_def:.2f}, Prec={p_def:.2f}')
    plt.scatter([r_tune], [p_tune], color='green', s=70, zorder=5, label=f'Val-tuned (thresh={chosen_thresh:.2f}): Rec={r_tune:.2f}, Prec={p_tune:.2f}')
    plt.title('V-Class Precision-Recall on DS2 (Threshold Chosen on Validation)')
    plt.xlabel('Recall (Ventricular Beats)')
    plt.ylabel('Precision (Ventricular Beats)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PR curve to {save_path}")
    print(f"  V-Class Default (0.50): Recall = {r_def:.4f}, Precision = {p_def:.4f}")
    print(f"  V-Class Tuned   ({chosen_thresh:.2f}): Recall = {r_tune:.4f}, Precision = {p_tune:.4f}")


def plot_attention_profiles(model, X_test, rr_test, y_test, save_path='results/attention_profiles.png'):
    print("Computing Multi-head Attention profiles...")
    model.eval()
    all_weights_per_class = {c: [] for c in range(5)}
    with torch.no_grad():
        for c in range(5):
            c_indices = np.where(y_test == c)[0][:80]
            if len(c_indices) == 0:
                continue
            bx = torch.tensor(X_test[c_indices], dtype=torch.float32).unsqueeze(1).to(DEVICE)
            brr = torch.tensor(rr_test[c_indices], dtype=torch.float32).to(DEVICE)
            _, weights = model(bx, brr) # (batch, seq, seq)
            # Average attention received by each temporal token across heads/batch
            mean_token_attn = weights.mean(dim=1).mean(dim=0).cpu().numpy() # (seq,)
            all_weights_per_class[c] = mean_token_attn

    time_axis = (np.arange(252) - 90) / 360.0 # Time relative to R-peak in seconds

    plt.figure(figsize=(8, 4))
    for c in range(5):
        if len(all_weights_per_class[c]) > 0:
            w = all_weights_per_class[c]
            # Interpolate to 252 for plotting alongside beat
            w_up = np.interp(np.linspace(0, 1, 252), np.linspace(0, 1, len(w)), w)
            plt.plot(time_axis, w_up, label=f"Class {CLASS_NAMES[c]}", lw=1.8)

    plt.axvline(0, color='gray', linestyle='--', alpha=0.5, label='R-peak')
    plt.title('Average Self-Attention Temporal Distribution by Beat Class')
    plt.xlabel('Time relative to R-peak (seconds)')
    plt.ylabel('Mean Attention Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved attention profiles to {save_path}")


def analyze_confidence_threshold(y_true, probs, save_path='results/confidence_vs_error.png'):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    is_correct = (preds == y_true)

    # Discretize confidence into bins
    bins = np.linspace(0.2, 1.0, 9)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_accs = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        bin_counts.append(np.sum(mask))
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(is_correct[mask]))
        else:
            bin_accs.append(np.nan)

    # Let's inspect threshold = 0.70
    flag_threshold = 0.70
    flagged_mask = confidences < flag_threshold
    unflagged_mask = confidences >= flag_threshold

    frac_flagged = np.mean(flagged_mask)
    err_flagged = 1.0 - np.mean(is_correct[flagged_mask]) if np.sum(flagged_mask) > 0 else 0.0
    err_unflagged = 1.0 - np.mean(is_correct[unflagged_mask])

    plt.figure(figsize=(7, 5))
    plt.plot(bin_centers, bin_accs, marker='o', lw=2, color='navy', label='Empirical Accuracy')
    plt.plot([0.2, 1.0], [0.2, 1.0], linestyle='--', color='gray', label='Perfect Calibration')
    plt.axvline(flag_threshold, color='crimson', linestyle=':', lw=2, label=f'Review Threshold ({flag_threshold})')
    plt.title('Confidence Calibration & Review Triage on DS2')
    plt.xlabel('Softmax Confidence (Max Probability)')
    plt.ylabel('Empirical Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confidence analysis to {save_path}")
    print(f"Confidence Threshold: {flag_threshold}")
    print(f"  Flagged for review: {frac_flagged*100:.1f}% of total test beats")
    print(f"  Error rate in flagged beats:   {err_flagged*100:.1f}%")
    print(f"  Error rate in unflagged beats: {err_unflagged*100:.1f}%")


def run_full_evaluation():
    print(" MIT-BIH COMPREHENSIVE EVALUATION & INTERPRETABILITY ")


    # Load data
    X, rr, y, rec = build_dataset()
    train_mask, val_mask, test_mask = get_splits(rec, y)

    X_val, rr_val, y_val = X[val_mask], rr[val_mask], y[val_mask]
    X_test, rr_test, y_test = X[test_mask], rr[test_mask], y[test_mask]

    # Check if training results exist
    results_csv = 'results/training_results.csv'
    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        summary = df_results[df_results['seed'].isin(SEEDS)].groupby('model').agg({
            'macro_f1': ['mean', 'std'],
            'v_recall': ['mean', 'std'],
            'v_precision': ['mean', 'std'],
            'avg_cost': ['mean', 'std']
        })
        summary.to_csv('results/summary_metrics.csv')
        print("Generated summary table saved to results/summary_metrics.csv")
    else:
        print("Note: results/training_results.csv not found yet. Run train.py first to populate training metrics.")

    # Load best ECGAttentionCNN model (seed 42)
    best_model_path = 'results/ecg_attention_seed42.pt'
    if not os.path.exists(best_model_path):
        print(f"Model checkpoint {best_model_path} not found. Please train models first using python train.py.")
        return

    print(f"Loading {best_model_path} for in-depth diagnostics...")
    model = load_model(ECGAttentionCNN, best_model_path)

    val_preds, val_probs = get_model_predictions_and_probs(model, X_val, rr_val)
    test_preds, test_probs = get_model_predictions_and_probs(model, X_test, rr_test)

    # 1. Confusion Matrix
    plot_confusion_matrix(y_test, test_preds)

    # 2. PR Curve with threshold tuned on validation
    evaluate_pr_curve(val_probs, y_val, test_probs, y_test)

    # 3. Attention profiles
    plot_attention_profiles(model, X_test, rr_test, y_test)

    # 4. Confidence analysis
    analyze_confidence_threshold(y_test, test_probs)


if __name__ == '__main__':
    run_full_evaluation()

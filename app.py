import os
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch
import torch.nn.functional as F

from data import filter_signal, cut_beats, build_dataset, get_splits, COST_MATRIX, CLASS_NAMES, DS2_RECS
from models import PlainCNN, ECGAttentionCNN

# Set page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Beat Classifier",
    page_icon="🫀",
    layout="wide"
)

# Custom student CSS styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.2rem; }
    .disclaimer-box { background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 4px; margin-bottom: 20px; color: #991b1b; font-size: 0.95rem; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
    .flagged-box { background-color: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px; margin-top: 10px; color: #92400e; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

CONFIDENCE_REVIEW_THRESHOLD = 0.70
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_app_models():
    plain_path = 'results/plain_cnn_seed42.pt'
    attn_path = 'results/ecg_attention_seed42.pt'

    plain_model = PlainCNN(num_classes=5)
    attn_model = ECGAttentionCNN(num_classes=5)

    plain_loaded = False
    attn_loaded = False

    if os.path.exists(plain_path):
        plain_model.load_state_dict(torch.load(plain_path, map_location=DEVICE, weights_only=True))
        plain_loaded = True

    if os.path.exists(attn_path):
        attn_model.load_state_dict(torch.load(attn_path, map_location=DEVICE, weights_only=True))
        attn_loaded = True

    plain_model.to(DEVICE).eval()
    attn_model.to(DEVICE).eval()

    return plain_model, attn_model, plain_loaded, attn_loaded


@st.cache_data
def load_dataset_cached():
    if os.path.exists('data_cache.npz'):
        data = np.load('data_cache.npz')
        X, rr, y, rec = data['X'], data['rr'], data['y'], data['rec']
        train_mask, val_mask, test_mask = get_splits(rec)
        # Class frequencies in DS1 train
        y_train = y[train_mask]
        train_counts = np.bincount(y_train, minlength=5)
        train_freqs = train_counts / (np.sum(train_counts) + 1e-8)
        return X, rr, y, rec, train_freqs
    return None, None, None, None, None


# App Header
st.markdown('<div class="main-title">🫀 MIT-BIH Arrhythmia Beat Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="disclaimer-box">⚠️ <b>Student Course Project Disclaimer:</b> This application and model are built strictly for educational and academic exploration. It is NOT certified, verified, or intended for real-world clinical diagnosis.</div>', unsafe_allow_html=True)

# Load data and models
X_all, rr_all, y_all, rec_all, ds1_train_freqs = load_dataset_cached()
plain_model, attn_model, plain_loaded, attn_loaded = load_app_models()

if ds1_train_freqs is None:
    # Fallback standard train class frequencies if cache not built yet
    ds1_train_freqs = np.array([0.89, 0.03, 0.06, 0.015, 0.005])

# Model Selection (main page)
model_choice = st.radio(
    "Active Classifier",
    ["ECGAttentionCNN (Conv + Self-Attention)", "Plain CNN (Baseline Conv1D)"],
    horizontal=True
)
active_model = attn_model if "ECGAttention" in model_choice else plain_model

if not attn_loaded and not plain_loaded:
    st.warning("⚠️ No `.pt` weights found in `results/`. Please run `python train.py` first.")

# Main Input Section
input_mode = st.radio(
    "Select Input Source",
    ["DS2 Inter-Patient Test Record", "Upload a Raw ECG Segment (.csv / .txt)"],
    horizontal=True
)

current_beat = None
current_rr = None
true_label = None
record_beats = []
record_rr_list = []
record_labels = []

if input_mode == "DS2 Inter-Patient Test Record":
    if X_all is None:
        st.error("Dataset cache `data_cache.npz` not found. Please run `python train.py` or `python data.py` first to generate it.")
        st.stop()

    # Filter to DS2 records
    available_ds2 = [int(r) for r in sorted(list(set(rec_all[np.isin(rec_all, DS2_RECS)])))]

    col_rec, col_idx, col_ctrl = st.columns([1.5, 2, 1.5])
    with col_rec:
        selected_rec = st.selectbox("DS2 Patient Record", available_ds2, index=0)

    # Filter beats for selected record
    rec_mask = (rec_all == selected_rec)
    rec_indices = np.where(rec_mask)[0]
    total_rec_beats = len(rec_indices)

    # Session state management for beat selection
    if 'beat_idx' not in st.session_state:
        st.session_state.beat_idx = 0

    with col_ctrl:
        st.write("")
        st.write("")
        c1, c2 = st.columns(2)
        if c1.button("🎲 Random"):
            st.session_state.beat_idx = random.randint(0, total_rec_beats - 1)

        if c2.button("🔍 Find Low Conf"):
            # Scan beats in current record to find lowest confidence beat
            with st.spinner("Scanning record for lowest confidence beat..."):
                test_x_t = torch.tensor(X_all[rec_indices], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                test_rr_t = torch.tensor(rr_all[rec_indices], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    out = active_model(test_x_t, test_rr_t)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    confs = np.max(probs, axis=1)
                lowest_idx = int(np.argmin(confs))
                st.session_state.beat_idx = lowest_idx

    with col_idx:
        # Keep index in bounds
        st.session_state.beat_idx = min(st.session_state.beat_idx, total_rec_beats - 1)
        selected_beat_pos = st.slider(
            f"Beat Index in Record {selected_rec} (Total: {total_rec_beats})",
            0, total_rec_beats - 1, st.session_state.beat_idx,
            key='beat_slider'
        )
        st.session_state.beat_idx = selected_beat_pos

    global_idx = rec_indices[st.session_state.beat_idx]
    current_beat = X_all[global_idx]
    current_rr = rr_all[global_idx]
    true_label = int(y_all[global_idx])

else:
    # Upload segment mode
    uploaded_file = st.file_uploader(
        "Upload raw ECG signal (1-column CSV or TXT, sampled at 360 Hz, min 5 seconds)",
        type=['csv', 'txt']
    )

    if uploaded_file is not None:
        try:
            # Read 1D column
            lines = uploaded_file.getvalue().decode('utf-8').strip().split()
            raw_floats = [float(x.replace(',', '')) for x in lines if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
            raw_sig = np.array(raw_floats)

            if len(raw_sig) < 360 * 5:
                st.error(f"Uploaded signal has only {len(raw_sig)} samples ({len(raw_sig)/360.0:.1f}s). Please provide at least 5 seconds (~1800 samples) at 360 Hz.")
                st.stop()

            # Preprocess using shared data.py routines (do not rewrite!)
            filt_sig = filter_signal(raw_sig, fs=360)
            # Find R-peaks
            peaks, _ = find_peaks(filt_sig, distance=180, prominence=0.4)

            if len(peaks) < 3:
                st.error("Could not reliably detect at least 3 R-peaks in the uploaded signal. Try adjusting the signal.")
                st.stop()

            beats, rr, keep_mask = cut_beats(filt_sig, peaks, fs=360)
            if len(beats) == 0:
                st.error("No valid beats could be segmented (window boundary check failed).")
                st.stop()

            st.success(f"Detected {len(beats)} valid ECG beats in uploaded segment!")

            if 'upload_beat_idx' not in st.session_state:
                st.session_state.upload_beat_idx = 0

            col_up1, col_up2 = st.columns([3, 1])
            with col_up2:
                if st.button("🔍 Find Low Conf Beat"):
                    with st.spinner("Analyzing uploaded beats..."):
                        bx_t = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(DEVICE)
                        brr_t = torch.tensor(rr, dtype=torch.float32).to(DEVICE)
                        with torch.no_grad():
                            out = active_model(bx_t, brr_t)
                            logits = out[0] if isinstance(out, tuple) else out
                            probs = F.softmax(logits, dim=-1).cpu().numpy()
                            confs = np.max(probs, axis=1)
                        st.session_state.upload_beat_idx = int(np.argmin(confs))

            with col_up1:
                st.session_state.upload_beat_idx = min(st.session_state.upload_beat_idx, len(beats) - 1)
                selected_beat_idx = st.slider(
                    f"Select Beat (1 to {len(beats)})",
                    0, len(beats) - 1, st.session_state.upload_beat_idx
                )
                st.session_state.upload_beat_idx = selected_beat_idx

            current_beat = beats[st.session_state.upload_beat_idx]
            current_rr = rr[st.session_state.upload_beat_idx]
            true_label = None # Uploaded beats have no ground truth label

        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")
            st.stop()
    else:
        st.info("Upload an ECG segment file above or switch to 'DS2 Inter-Patient Test Record' mode.")
        st.stop()

# Beat Analysis and Prediction
if current_beat is not None:
    st.markdown("---")

    # Prepare inputs for PyTorch
    bx = torch.tensor(current_beat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    brr = torch.tensor(current_rr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Get model probabilities and optional attention weights
    with torch.no_grad():
        out = active_model(bx, brr)
        if isinstance(out, tuple):
            logits, attn_weights = out
            attn_weights_np = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()
        else:
            logits = out
            attn_weights_np = None

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_class = int(np.argmax(probs))
        pred_confidence = probs[pred_class]

    # Display Metrics & Classification
    col_plot, col_info = st.columns([1.8, 1.2])

    with col_plot:
        st.subheader("ECG Beat Waveform")
        time_axis = (np.arange(252) - 90) / 360.0

        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(time_axis, current_beat, color='#1e293b', lw=2, label='Filtered Beat Waveform')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.6, label='R-peak (t=0)')
        ax.axvspan(-0.08, 0.08, color='#38bdf8', alpha=0.15, label='QRS Zone (±80ms)')
        ax.set_xlabel("Time relative to R-peak (seconds)", fontsize=10)
        ax.set_ylabel("Normalized Amplitude", fontsize=10)
        ax.set_title(f"Beat waveform (Predicted class: {CLASS_NAMES[pred_class]})", fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        st.pyplot(fig)
        plt.close()

        # Optional Attention plot
        if attn_weights_np is not None:
            with st.expander("👁️ View Multi-Head Attention Temporal Weights"):
                fig_attn, ax_attn = plt.subplots(figsize=(8, 2.5))
                w_up = np.interp(np.linspace(0, 1, 252), np.linspace(0, 1, len(attn_weights_np)), attn_weights_np)
                ax_attn.plot(time_axis, w_up, color='#4338ca', lw=1.8, label='Attention Weight')
                ax_attn.axvline(0, color='gray', linestyle='--', alpha=0.5)
                ax_attn.set_xlabel("Time relative to R-peak (seconds)", fontsize=9)
                ax_attn.set_ylabel("Weight", fontsize=9)
                ax_attn.set_title("Self-Attention temporal focus across beat sequence", fontsize=10)
                st.pyplot(fig_attn)
                plt.close()

    with col_info:
        st.subheader("Classification Result")

        pred_name = CLASS_NAMES[pred_class]
        st.markdown(f"### Predicted Class: **{pred_name}** ({pred_class})")
        st.markdown(f"**Confidence:** `{pred_confidence*100:.1f}%`")
        st.markdown(f"**R-R intervals:** Prev = `{current_rr[0]:.2f}s`, Next = `{current_rr[1]:.2f}s`")

        if true_label is not None:
            true_name = CLASS_NAMES[true_label]
            is_correct = (pred_class == true_label)
            status_color = "green" if is_correct else "red"
            st.markdown(f"**Ground Truth Label:** :{status_color}[**{true_name}**] ({'Match ✅' if is_correct else 'Error ❌'})")

        # Flag for manual review if low confidence
        if pred_confidence < CONFIDENCE_REVIEW_THRESHOLD:
            st.markdown(f"""
            <div class="flagged-box">
                ⚠️ <b>Flagged for Manual Review:</b><br>
                Model confidence ({pred_confidence*100:.1f}%) is below safety threshold ({CONFIDENCE_REVIEW_THRESHOLD*100:.0f}%).
                This beat should be triaged for expert cardiologist inspection.
            </div>
            """, unsafe_allow_html=True)

        # Softmax distribution bar chart
        st.markdown("#### Class Probabilities")
        prob_df = pd.DataFrame({
            'Class': CLASS_NAMES,
            'Probability': probs
        })
        st.bar_chart(prob_df.set_index('Class'), height=150)

        # -------------------------------------------------------------
        # Clinical Cost of Being Wrong Section
        # -------------------------------------------------------------
        st.markdown("---")
        st.subheader(" Clinical Cost Analysis")
        p = pred_class

        # 1. Expected cost if wrong: weighted by DS1 train class frequencies (excluding true==p)
        other_classes = [c for c in range(5) if c != p]
        other_weights = np.array([ds1_train_freqs[c] for c in other_classes])
        if np.sum(other_weights) > 0:
            norm_other_weights = other_weights / np.sum(other_weights)
            costs_if_wrong = np.array([COST_MATRIX[c, p] for c in other_classes])
            expected_cost_wrong = float(np.sum(norm_other_weights * costs_if_wrong))
        else:
            expected_cost_wrong = 0.0

        # 2. Worst-case cost: max over true classes and which true class it comes from
        worst_case_cost = -1
        worst_true_class = None
        for c in range(5):
            if c != p and COST_MATRIX[c, p] > worst_case_cost:
                worst_case_cost = COST_MATRIX[c, p]
                worst_true_class = CLASS_NAMES[c]

        # 3. Actual cost (for DS2 beats only)
        if true_label is not None:
            actual_cost = COST_MATRIX[true_label, p]

        st.markdown(f"""
        - **Expected cost if wrong:** `{expected_cost_wrong:.2f}`  
          *(Weighted by DS1 baseline class frequencies if true class != {pred_name})*
        - **Worst-case cost:** `{worst_case_cost}`  
          *(If beat was actually **{worst_true_class}**, e.g. a missed life-threatening arrhythmia)*
        """)

        if true_label is not None:
            actual_color = "green" if actual_cost == 0 else "red"
            st.markdown(f"- **Actual cost for this beat:** :{actual_color}[`{actual_cost}`] *(True={true_name}, Pred={pred_name})*")
        else:
            st.caption("*(Actual cost unavailable for uploaded beats without clinician annotations)*")
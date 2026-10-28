# Heart Arrhythmia Beat Classification on MIT-BIH

Beat-level arrhythmia classification (N, S, V, F, Q) on the MIT-BIH Arrhythmia Database, built around the part most tutorials skip: **evaluating on new patients** and **measuring what mistakes cost**. I compare an RR-interval Random Forest, a plain 1D CNN, a parameter-matched wide CNN, and a CNN with multi-head self-attention, all trained with the same setup over 3 seeds.

> Student project, not for clinical use.

---

## Highlights

- **Clinically standard evaluation.** Strict inter-patient split (de Chazal DS1 train / DS2 test, split by record), so scores reflect generalization to unseen patients rather than memorized patient morphology.
- **Leakage demonstrated, not just claimed.** A naive random beat split is run side by side with the inter-patient split using the same CNN and training setup. Accuracy drops from **97.6% to 86.1%** and 4-class macro F1 from **0.850 to 0.396** once beats from the same patient can't appear in both train and test.
- **Controlled model comparison.** Four models, identical training setup, 3 seeds, mean ± std, plus a parameter-matched wide CNN so any gain from attention isn't just extra parameters.
- **Cost-sensitive evaluation.** A clinically motivated cost matrix and an "always predict Normal" baseline, so every model is judged by what its errors cost, not only by F1.
- **Confidence-based flagging.** Low-confidence beats are flagged for review in a Streamlit demo.

---

## Leakage test: random split vs inter-patient split

Same plain CNN, same class weights, same early stopping, 3 seeds. The naive split pools all 44 records and splits beats randomly, with the same train/val/test sizes as the inter-patient split, so the only difference is whether a patient's beats can be on both sides.

| Split | Accuracy | Macro F1 (5-class) | Macro F1 (4-class: N, S, V, F) |
|---|---|---|---|
| Naive random beat split (leaky) | 0.976 ± 0.003 | 0.680 ± 0.010 | 0.850 ± 0.012 |
| Inter-patient (DS1 -> DS2) | 0.861 ± 0.039 | 0.317 ± 0.039 | 0.396 ± 0.048 |

Accuracy falls about 12 points and 4-class macro F1 falls by more than half. The naive number looks like the "98% accuracy" in many ECG tutorials, but it mostly measures how well the model recognizes patients it has already seen. Run it yourself with `python leakage_test.py`.

---

## Results (DS2 test set, mean ± std over 3 seeds)

| Model | Macro F1 (5-class) | V recall | V precision | Avg cost / beat |
|---|---|---|---|---|
| All-N baseline (always predict Normal) | 0.188 | 0.000 | n/a | 0.422 |
| Random Forest (RR only) | 0.320 ± 0.001 | 0.704 ± 0.001 | 0.404 ± 0.004 | 0.425 ± 0.003 |
| Plain CNN | 0.330 ± 0.021 | 0.799 ± 0.021 | 0.579 ± 0.060 | 0.403 ± 0.128 |
| CNN + self-attention | 0.323 ± 0.015 | 0.790 ± 0.044 | 0.539 ± 0.104 | 0.374 ± 0.032 |
| Wide CNN (matched params) | **0.352 ± 0.030** | **0.861 ± 0.036** | **0.686 ± 0.115** | 0.451 ± 0.178 |
| CNN + attention, unweighted loss (1 seed) | 0.347 | 0.845 | 0.691 | **0.270** |

The All-N row is computed from the DS2 class counts (N 44218, S 1836, V 3219, F 388, Q 7) and the cost matrix below.

**What the results show**

- **Waveform models beat timing alone.** Every CNN variant has clearly higher V recall (0.79 to 0.86) than the RR-only Random Forest (0.70), so the beat shape carries information beyond RR timing.
- **Wide CNN is the strongest on F1 and V detection** (macro F1 0.352, V recall 0.861, V precision 0.686).
- **Attention matches the plain CNN.** Macro F1 (0.323 vs 0.330) and V recall (0.790 vs 0.799) differ by less than the seed std, so the two are statistically indistinguishable here. Attention has the lowest cost among the class-weighted neural models (0.374) and a much tighter cost spread across seeds (std 0.032 vs 0.128 for the plain CNN).
- **Beating the trivial baseline on cost.** Attention (0.374), the plain CNN (0.403) and the unweighted attention run (0.270) all beat the All-N baseline (0.422). The unweighted run is about 36% cheaper than All-N, though it is a single seed.
- **Macro F1 is held down by class Q.** DS1 has only 4 Q beats and DS2 has 7, so Q's F1 is near zero for every model. The N/S/V/F scores are the more meaningful picture.

### Cost matrix

Rows are the true class, columns the predicted class, order N, S, V, F, Q:

```python
COST_MATRIX = [
    [0, 1, 3, 2, 1],  # true N
    [2, 0, 3, 2, 1],  # true S
    [5, 4, 0, 3, 2],  # true V  (missing a V as N costs the most)
    [3, 2, 4, 0, 2],  # true F
    [1, 1, 2, 1, 0],  # true Q
]
```

Missing a ventricular beat is penalized most; a false alarm on a normal beat costs little. The values are illustrative.

### Confidence flagging

Using softmax confidence with a 0.7 threshold on DS2 (earlier run, attention model): **38.7% of beats were flagged**, with a **44.8% error rate among flagged beats versus 26.2% among unflagged beats**, so flagged beats are about 1.7x more likely to be wrong. The demo app shows this as a "flagged for manual review" warning.

---

## Streamlit app

Browse DS2 beats or upload a raw ECG segment (360 Hz). The app shows the beat waveform, the predicted class, the model's confidence, and the cost of being wrong, and it flags low-confidence beats for review.

![Confident beat](/screenshot_normal.png)
*A confident prediction.*

![Flagged beat](/screenshot_flagged.png)
*A low-confidence beat flagged for manual review.*

---

## How to run

```bash
pip install -r requirements.txt
python train.py        # trains all models over 3 seeds, saves .pt files and results csv
python evaluate.py     # tables, PR curve, confusion matrix, confidence plots
python leakage_test.py # naive random split vs inter-patient split
streamlit run app.py   # demo app
```


---

## Project structure

```
data.py          load, filter, segment, split, cache to .npz
models.py        Random Forest, plain CNN, wide CNN, CNN + attention
train.py         trains all models, saves .pt and results csv
evaluate.py      tables, cost analysis, PR curve, confidence plots
leakage_test.py  naive random split vs inter-patient split
app.py           Streamlit demo
results/         saved tables, plots, model files
```

---

## Scope and limitations

- Single dataset (44 MIT-BIH records), with beats cut around annotated R-peaks.
- Rare classes are tiny (Q: 4 train / 7 test beats), so Q and F results are noisy.
- Validation is 4 records, so early stopping is noisy.
- Seed variance is large for several metrics (cost std up to 0.178), and the unweighted row is one seed, so model differences are mostly within noise.
- Some training choices (class weights, scheduler, early stopping) were adjusted after looking at DS2 results, so DS2 numbers may be slightly optimistic.
- The cost matrix is illustrative, not clinically validated.

## Next steps

- Rerun the unweighted variant with 3 seeds.
- Report 4-class macro F1 (N, S, V, F).
- Pick the V decision threshold on validation only.
- Calibrate confidence (temperature scaling) and flag with a fixed review budget.
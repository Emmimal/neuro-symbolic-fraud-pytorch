# Hybrid Neuro-Symbolic Fraud Detection

A PyTorch implementation of a hybrid neuro-symbolic training objective for credit card fraud detection on severely imbalanced data. Domain knowledge is injected directly into the loss function as a differentiable rule penalty — encoding analyst intuition about suspicious transaction patterns without requiring additional labels.

---

## Overview

Standard neural networks trained on imbalanced fraud datasets optimize for overall loss reduction, which means predicting "not fraud" on uncertain transactions is almost always rewarded. The model has no built-in concept of what fraud *looks like* in feature space.

This project adds a differentiable rule loss to the training objective that penalizes the model for assigning low fraud probability to transactions that exhibit two domain-known signals of suspicion: unusually high transaction amount and an atypical PCA signature. The rule fires on every batch regardless of whether labeled fraud appears in it — providing gradient signal that BCE alone cannot.

```
L_total = L_BCE + λ · L_rule
```

The experiment compares four approaches on the Kaggle Credit Card Fraud dataset:

| Model | F1 (seed=42) | PR-AUC | ROC-AUC | Recall@1%FPR |
|---|---|---|---|---|
| Isolation Forest | 0.121 | 0.172 | 0.941 | 0.581 |
| One-Class SVM | 0.029 | 0.391 | 0.930 | 0.797 |
| Pure Neural (λ=0) | 0.776 | 0.806 | 0.969 | 0.878 |
| **Hybrid (λ=0.5)** | **0.767** | **0.745** | **0.970** | **0.878** |

Multi-seed variance across 5 seeds ([42, 0, 7, 123, 2024]):

| Model | F1 (mean ± std) | PR-AUC (mean ± std) | ROC-AUC (mean ± std) |
|---|---|---|---|
| Pure Neural | 0.783 ± 0.024 | 0.760 ± 0.026 | 0.967 ± 0.003 |
| **Hybrid (λ=0.5)** | **0.774 ± 0.027** | **0.740 ± 0.058** | **0.970 ± 0.005** |

> The hybrid achieves consistently higher ROC-AUC across all 5 seeds. ROC-AUC is threshold-independent — it measures ranking quality across all possible cutoffs and is the cleanest signal from this experiment. F1 and PR-AUC differences are within seed variance.

---

## The Rule Loss

```python
def rule_loss(x, probs):
    # x[:, -1]   = Amount  (last column in creditcard.csv after dropping Class)
    # x[:, 1:29] = V1-V28  (PCA components, columns 1-28)
    amount   = x[:, -1]
    pca_norm = torch.norm(x[:, 1:29], dim=1)

    suspicious = (
        torch.sigmoid(5 * (amount   - amount.mean())) +
        torch.sigmoid(5 * (pca_norm - pca_norm.mean()))
    ) / 2.0

    penalty = suspicious * torch.relu(0.6 - probs.squeeze())
    return penalty.mean()
```

**Why sigmoid instead of a hard threshold?**
A step function has zero gradient everywhere except at the threshold itself, where it is undefined. Backpropagation has nothing to work with. The steep sigmoid (slope=5, centered at the batch mean) approximates the same threshold behavior while remaining smooth and differentiable — the gradient peaks near the boundary and is small far from it.

**Why PCA norm?**
V1–V28 are PCA-transformed features from the original anonymized transaction space. A transaction that sits far from the origin in this compressed space has unusual variance across multiple original features simultaneously. The Euclidean norm captures that distance in a single scalar. On non-PCA data, substitute a domain-appropriate anomaly signal: Mahalanobis distance, isolation score, or a feature-specific z-score.

**Why one-sided relu?**
`relu(0.6 - probs)` fires only when the predicted fraud probability is *below* 0.6 for a suspicious transaction. If the model is already confident (prob > 0.6), the penalty is zero. The rule can never fight against a correct high-confidence prediction — only push up underconfident ones.

---

## Project Structure

```
neuro-symbolic-fraud-pytorch/
├── fraud_hybrid.py                  # Full experiment: baselines, lambda sweep, evaluation, plots
├── creditcard.csv          # Not included — download from Kaggle (see below)
├── requirements.txt
└── README.md
```

**Generated outputs after running `app.py`:**

```
pr_curve.png               # Precision-Recall curve, hybrid model, seed=42
confusion_matrix.png       # Confusion matrix at val-tuned threshold, seed=42
probability_histogram.png  # Score distributions, fraud vs non-fraud
variance_analysis.png      # F1 and PR-AUC mean ± std across 5 seeds
```

---

## Setup

### Requirements

```
python >= 3.9
torch >= 2.0
scikit-learn >= 1.3
pandas
numpy
matplotlib
seaborn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

Download `creditcard.csv` from Kaggle:
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place it in the same directory as `fraud_hybrid.py`. The file is not included in this repo due to size.

Column layout assumed by the code:

```
Time | V1 | V2 | ... | V28 | Amount | Class
  0     1    2          28     29       30
```

- `x[:, -1]` → Amount (index 29 after dropping Class)
- `x[:, 1:29]` → V1–V28 PCA components

---

## Running the Experiment

```bash
python app.py
```

This will:

1. Train Isolation Forest and One-Class SVM baselines
2. Run a lambda sweep (λ ∈ [0.0, 0.1, 0.5, 1.0, 2.0]) on seed=42 to find the best hybrid configuration
3. Evaluate the best hybrid and a pure neural baseline on the test set (seed=42)
4. Run multi-seed variance analysis across 5 seeds for both models
5. Save four plots to the working directory

**Expected runtime:** 15–40 minutes depending on hardware. The One-Class SVM on 199k samples is the slowest step. A GPU will significantly speed up the neural model training.

---

## Key Design Decisions

### Data Split

70 / 15 / 15 train/val/test, stratified on the fraud label. The val set is used exclusively for:

- Early stopping (monitored metric: val PR-AUC, patience=7)
- Lambda selection
- Threshold tuning

The test set is never seen during any of these steps.

### Threshold Evaluation

Both the hybrid and pure neural baseline use val-optimized thresholds applied to the test set:

```python
def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return thresholds[np.argmax(f1_scores)]
```

On a 0.17% positive-rate dataset the optimal F1 threshold is nowhere near 0.5. Applying different thresholding strategies to different models means measuring the threshold gap, not the model gap. Both models use identical evaluation logic throughout.

### Class Weighting

```python
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~577
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

A single fraud sample generates ~577x the gradient of a non-fraud sample. This is the baseline before the rule loss is introduced.

### Batch Size

`BATCH_SIZE = 2048` is intentional. Larger batches stabilize the batch-relative sigmoid thresholds in the rule loss — the amount mean and PCA norm mean are computed per batch, so small batches cause the "suspicious" boundary to shift significantly between steps.

---

## Caveats and Known Limitations

**Batch-relative statistics in deployment.**
The rule computes suspicion relative to the *batch* mean, not a fixed population statistic. During training this is stable due to large batch size and stratified sampling. In online inference (scoring individual transactions), freeze the amount mean and PCA norm mean to training-set statistics before deploying. Otherwise the suspicious boundary shifts with every inference call.

**PR-AUC variance increases with the rule loss.**
Hybrid PR-AUC ranges from 0.636 to 0.817 across seeds vs 0.731 to 0.806 for the pure baseline. The rule loss makes some initializations better and some worse. Multi-seed validation is essential before drawing conclusions from any single run.

**Lambda sensitivity.**
λ=1.0 and 2.0 show a meaningful drop in validation PR-AUC compared to λ=0.5. Aggressive rule weighting can override the BCE signal rather than complement it. Start at λ=0.5 and verify on your data before increasing.

**No gradient boosting comparison.**
A tuned XGBoost or LightGBM model would likely outperform both neural approaches on this dataset. The comparison here is designed to isolate the effect of the rule loss, not to benchmark against all possible methods.

---

## Extending the Rule Loss

### Learnable Combination Weights

The current implementation weights amount and PCA norm equally (0.5 / 0.5). A natural extension makes these learnable:

```python
# In MLP.__init__:
self.rule_w = nn.Parameter(torch.tensor([0.5, 0.5]))

# In rule_loss:
w = torch.softmax(self.rule_w, dim=0)
suspicious = (
    w[0] * torch.sigmoid(5 * (amount   - amount.mean())) +
    w[1] * torch.sigmoid(5 * (pca_norm - pca_norm.mean()))
)
```

This lets the model decide which signal is more predictive for the specific data distribution rather than hard-coding equal weights. Not yet benchmarked.

### Non-PCA Features

Replace `pca_norm` with any scalar that captures "distance from normal behavior" for your domain:

- **Mahalanobis distance** from the training-set centroid
- **Isolation score** from a fitted `IsolationForest`
- **Feature z-score** if a single raw feature is your primary anomaly signal

The rest of the rule loss structure remains identical.

---

## References

- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
  [https://dalpozz.github.io/static/pdf/SSCI_calib_final_noCC.pdf](https://dalpozz.github.io/static/pdf/SSCI_calib_final_noCC.pdf)

- ULB Machine Learning Group. *Credit Card Fraud Detection Dataset* (Kaggle).
  [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. arXiv:1502.03167.
  [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

- PyTorch Documentation — BCEWithLogitsLoss.
  [https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

---

## License

MIT License. See `LICENSE` for details.

---

## Author

[@Emmimal](https://github.com/Emmimal)

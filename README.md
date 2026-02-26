# neuro-symbolic-fraud-pytorch
Hybrid neuro-symbolic fraud detector: soft rule penalties in loss improve F1 by 123% on Kaggle Credit Card Fraud dataset.

# Neuro-Symbolic Fraud Detection Experiment

A PyTorch implementation showing how to inject simple, differentiable domain rules into a neural network's loss function to improve fraud detection on highly imbalanced transaction data.

By adding a soft rule-based penalty term, the model learns to assign higher fraud probabilities to suspicious-looking transactions (high amount + unusual PCA magnitude), resulting in significantly better F1-score and PR-AUC compared to a pure neural baseline — without using oversampling, SMOTE, or complex architectures.

### Key Results (from the experiment run)

| Model                        | F1 (validation-tuned threshold) | PR-AUC | ROC-AUC | Recall @ 1% FPR |
|------------------------------|----------------------------------|--------|---------|-----------------|
| Isolation Forest             | 0.121                            | 0.172  | 0.941   | 0.581           |
| One-Class SVM                | 0.029                            | 0.391  | 0.930   | 0.797           |
| Pure Neural (λ = 0.0)        | 0.335                            | 0.638  | 0.966   | 0.865           |
| **Hybrid Neuro-Symbolic (λ = 2.0)** | **0.747**                 | **0.730** | **0.968** | **0.878**    |

- **F1 improvement** over pure neural baseline: **+123%** relative (0.335 → 0.747)
- **PR-AUC improvement**: **+14.4%** (0.638 → 0.730)
- Best λ (selected via validation PR-AUC): **2.0**
- Lambda ablation showed that very weak regularization (λ=0.1) actually hurt performance slightly compared to no rules.

Generated output plots (saved automatically when you run the script):

- `pr_curve.png` — Precision-Recall curve for the hybrid model  
- `confusion_matrix.png` — Confusion matrix at the tuned threshold  
- `probability_histogram.png` — Predicted probability distribution (fraud vs non-fraud)

### Dataset

- **Name**: Credit Card Fraud Detection  
- **Source**: https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv  
- **License**: ODbL / CC BY-SA 4.0 (commercial use allowed with attribution)  
- **Size**: 284,807 transactions, 492 frauds (0.172% positive rate)  
- **Features**: Time, V1–V28 (PCA-transformed), Amount, Class

### Requirements
```text
Python >= 3.8
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
```
**Install dependencies:**
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```
**Quick Start**

Download creditcard.csv from Kaggle and place it in the same folder as the script.

**Run the experiment:**
```python 
fraud_hybrid.py
```
(or open fraud_hybrid.ipynb in Jupyter/Colab)
The script will:

- Load and preprocess the data (standard scaling + stratified splits)
- Train the pure MLP and hybrid models across different λ values
- Perform lambda tuning using validation PR-AUC + early stopping
- Select the best model
- Evaluate on the test set using a threshold optimized on validation
- Generate and save the three diagnostic plots

Expected runtime: 5–15 minutes on CPU (faster with GPU).
**Core Implementation Highlights**
**Neural Backbone**
```Python
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
```
**Differentiable Rule Penalty**
```Python
def rule_loss(x, probs):
    amount   = x[:, -1]
    pca_norm = torch.norm(x[:, 1:29], dim=1)
    suspicious = (
        torch.sigmoid(5 * (amount   - amount.mean())) +
        torch.sigmoid(5 * (pca_norm - pca_norm.mean()))
    ) / 2.0
    penalty = suspicious * torch.relu(0.6 - probs.squeeze())
    return penalty.mean()
```
**Combined Loss (inside training loop)**
```Python
bce = criterion(logits.squeeze(), yb)
probs = torch.sigmoid(logits)
rl = rule_loss(xb, probs)
loss = bce + lambda_rule * rl
```
**Threshold Tuning (on validation set)**
```Python
def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    idx = np.argmax(f1_scores)
    return thresholds[idx]
```

**Why This Approach Helps**
On extreme imbalance (0.172% positives), pure neural models tend to become overly conservative — predicting almost everything as non-fraud. The rule penalty provides an additional, label-independent gradient signal on suspicious transactions, pushing fraud probabilities higher when domain heuristics indicate risk — even in batches without any labeled fraud.

**Limitations & Notes**

The rule uses batch-relative means for suspicion thresholds. In production, replace with fixed training-set statistics to avoid inference-time drift.
Only two simple rules are used here. You can extend with velocity, time-of-day, or learnable rule weights.
Threshold is tuned to maximize F1 on validation — in real systems, tune to your actual cost ratio (false negative vs false positive cost).

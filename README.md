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
- **Source**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
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

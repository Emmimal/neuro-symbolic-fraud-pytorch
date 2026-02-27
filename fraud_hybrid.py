import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# =============================
# CONFIG
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048
EPOCHS = 60
LR = 1e-3
LAMBDA_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0]
MAX_FPR = 0.01
EARLY_STOPPING_PATIENCE = 7
SEEDS = [42, 0, 7, 123, 2024]   # for variance analysis

# Fixed seed for data splitting — kept separate from model seeds
torch.manual_seed(42)
np.random.seed(42)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["Class"]).values
y = df["Class"].values

# NOTE: x[:, -1] = Amount, x[:, 1:29] = V1-V28 (PCA components)
# Column order assumed: Time, V1-V28, Amount (matches Kaggle creditcard.csv)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =============================
# MODEL
# =============================
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

    def forward(self, x):
        return self.net(x)

# =============================
# DIFFERENTIABLE RULE
# =============================
def rule_loss(x, probs):
    # x[:, -1]    = Amount (last column)
    # x[:, 1:29]  = V1-V28 PCA components
    amount   = x[:, -1]
    pca_norm = torch.norm(x[:, 1:29], dim=1)

    suspicious = (
        torch.sigmoid(5 * (amount   - amount.mean())) +
        torch.sigmoid(5 * (pca_norm - pca_norm.mean()))
    ) / 2.0

    penalty = suspicious * torch.relu(0.6 - probs.squeeze())
    return penalty.mean()

# =============================
# METRICS
# =============================
def recall_at_fpr(y_true, probs, max_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, probs)
    idx = np.where(fpr <= max_fpr)[0]
    return tpr[idx[-1]] if len(idx) > 0 else 0.0

def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    idx = np.argmax(f1_scores)
    return thresholds[idx], f1_scores[idx]

def evaluate(y_true, probs, threshold=0.5):
    preds = (probs > threshold).astype(int)
    return {
        "F1":           f1_score(y_true, preds),
        "PR-AUC":       average_precision_score(y_true, probs),
        "ROC-AUC":      roc_auc_score(y_true, probs),
        "Recall@1%FPR": recall_at_fpr(y_true, probs, MAX_FPR)
    }

# =============================
# TRAIN FUNCTION
# =============================
def train_model(lambda_rule, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(X_train.shape[1]).to(DEVICE)

    pos_weight = torch.tensor(
        (y_train == 0).sum() / (y_train == 1).sum(),
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_pr = 0
    patience    = 0
    best_state  = None

    for epoch in range(EPOCHS):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            bce    = criterion(logits.squeeze(), yb)
            probs  = torch.sigmoid(logits)

            rl   = rule_loss(xb, probs)
            loss = bce + lambda_rule * rl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).detach().cpu().numpy().flatten()
            val_pr    = average_precision_score(y_val, val_probs)

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state  = model.state_dict()
            patience    = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val_pr

# =============================
# UNSUPERVISED BASELINES
# =============================
print("Training Isolation Forest...")
iso = IsolationForest(contamination=0.001, random_state=42)
iso.fit(X_train)
iso_scores = -iso.decision_function(X_test)

print("Training One-Class SVM...")
svm = OneClassSVM(nu=0.001)
svm.fit(X_train[y_train == 0])
svm_scores = -svm.decision_function(X_test)

scaler_scores = MinMaxScaler()
iso_probs = scaler_scores.fit_transform(iso_scores.reshape(-1, 1)).flatten()
svm_probs = scaler_scores.fit_transform(svm_scores.reshape(-1, 1)).flatten()

iso_metrics = evaluate(y_test, iso_probs)
svm_metrics = evaluate(y_test, svm_probs)

# =============================
# LAMBDA TUNING  (seed=42)
# =============================
best_lambda    = None
best_model     = None
best_val_score = 0

print("\nTuning lambda_rule...")

for l in LAMBDA_VALUES:
    model, val_pr = train_model(l, seed=42)
    print(f"Lambda {l} → Val PR-AUC: {val_pr:.4f}")

    if val_pr > best_val_score:
        best_val_score = val_pr
        best_lambda    = l
        best_model     = model

print(f"\nBest Lambda: {best_lambda}")

# =============================
# FINAL TEST EVALUATION (seed=42)
# =============================
best_model.eval()

with torch.no_grad():
    val_probs_for_thresh = torch.sigmoid(best_model(X_val_t)).detach().cpu().numpy().flatten()
    test_probs           = torch.sigmoid(best_model(X_test_t)).detach().cpu().numpy().flatten()

best_thresh, _ = find_best_threshold(y_val, val_probs_for_thresh)
hybrid_metrics = evaluate(y_test, test_probs, threshold=best_thresh)

# Pure baseline (seed=42, single run — noted in paper)
pure_model, _ = train_model(0.0, seed=42)
pure_model.eval()

with torch.no_grad():
    pure_val_probs = torch.sigmoid(pure_model(X_val_t)).detach().cpu().numpy().flatten()
    pure_probs     = torch.sigmoid(pure_model(X_test_t)).detach().cpu().numpy().flatten()

pure_thresh, _ = find_best_threshold(y_val, pure_val_probs)
pure_metrics   = evaluate(y_test, pure_probs, threshold=pure_thresh)

# =============================
# PRINT SINGLE-SEED RESULTS
# =============================
print("\n--- FINAL RESULTS (seed=42) ---")
print("Isolation Forest:",      iso_metrics)
print("One-Class SVM:",         svm_metrics)
print("Pure Neural:",           pure_metrics)
print("Hybrid Neuro-Symbolic:", hybrid_metrics)

# =============================
# MULTI-SEED VARIANCE ANALYSIS
# =============================
print("\n--- MULTI-SEED VARIANCE ANALYSIS ---")
print(f"Running {len(SEEDS)} seeds: {SEEDS}\n")

hybrid_f1s,    pure_f1s    = [], []
hybrid_prauc,  pure_prauc  = [], []
hybrid_rocauc, pure_rocauc = [], []

for seed in SEEDS:
    # --- Hybrid ---
    h_model, _ = train_model(best_lambda, seed=seed)
    h_model.eval()
    with torch.no_grad():
        h_val_p  = torch.sigmoid(h_model(X_val_t)).cpu().numpy().flatten()
        h_test_p = torch.sigmoid(h_model(X_test_t)).cpu().numpy().flatten()
    h_thresh, _ = find_best_threshold(y_val, h_val_p)
    h_metrics   = evaluate(y_test, h_test_p, threshold=h_thresh)

    hybrid_f1s.append(h_metrics["F1"])
    hybrid_prauc.append(h_metrics["PR-AUC"])
    hybrid_rocauc.append(h_metrics["ROC-AUC"])

    # --- Pure baseline ---
    p_model, _ = train_model(0.0, seed=seed)
    p_model.eval()
    with torch.no_grad():
        p_val_p  = torch.sigmoid(p_model(X_val_t)).cpu().numpy().flatten()
        p_test_p = torch.sigmoid(p_model(X_test_t)).cpu().numpy().flatten()
    p_thresh, _ = find_best_threshold(y_val, p_val_p)
    p_metrics   = evaluate(y_test, p_test_p, threshold=p_thresh)

    pure_f1s.append(p_metrics["F1"])
    pure_prauc.append(p_metrics["PR-AUC"])
    pure_rocauc.append(p_metrics["ROC-AUC"])

    print(f"Seed {seed:>4} | Hybrid F1: {h_metrics['F1']:.3f}  PR-AUC: {h_metrics['PR-AUC']:.3f} "
          f"| Pure F1: {p_metrics['F1']:.3f}  PR-AUC: {p_metrics['PR-AUC']:.3f}")

print("\n--- VARIANCE SUMMARY ---")
print(f"Hybrid  F1:     {np.mean(hybrid_f1s):.3f} ± {np.std(hybrid_f1s):.3f}")
print(f"Pure    F1:     {np.mean(pure_f1s):.3f} ± {np.std(pure_f1s):.3f}")
print(f"Hybrid  PR-AUC: {np.mean(hybrid_prauc):.3f} ± {np.std(hybrid_prauc):.3f}")
print(f"Pure    PR-AUC: {np.mean(pure_prauc):.3f} ± {np.std(pure_prauc):.3f}")
print(f"Hybrid  ROC-AUC:{np.mean(hybrid_rocauc):.3f} ± {np.std(hybrid_rocauc):.3f}")
print(f"Pure    ROC-AUC:{np.mean(pure_rocauc):.3f} ± {np.std(pure_rocauc):.3f}")

# =============================
# VISUALIZATIONS
# =============================

# 1. PR Curve (best hybrid model, seed=42)
precision, recall, _ = precision_recall_curve(y_test, test_probs)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Hybrid, seed=42)")
plt.savefig("pr_curve.png")
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, (test_probs > best_thresh).astype(int))
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Hybrid (seed=42)")
plt.savefig("confusion_matrix.png")
plt.close()

# 3. Score Distribution
plt.figure()
plt.hist(test_probs[y_test == 0], bins=50, alpha=0.5, label="Non-Fraud")
plt.hist(test_probs[y_test == 1], bins=50, alpha=0.5, label="Fraud")
plt.legend()
plt.title("Fraud vs Non-Fraud Probability Distribution (Hybrid, seed=42)")
plt.savefig("probability_histogram.png")
plt.close()

# 4. Variance bar chart — F1 across seeds
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(["Pure Neural", "Hybrid"],
            [np.mean(pure_f1s), np.mean(hybrid_f1s)],
            yerr=[np.std(pure_f1s), np.std(hybrid_f1s)],
            capsize=8, color=["steelblue", "darkorange"], alpha=0.85)
axes[0].set_ylabel("F1 Score")
axes[0].set_title(f"F1 across {len(SEEDS)} seeds (mean ± std)")
axes[0].set_ylim(0, 1)

axes[1].bar(["Pure Neural", "Hybrid"],
            [np.mean(pure_prauc), np.mean(hybrid_prauc)],
            yerr=[np.std(pure_prauc), np.std(hybrid_prauc)],
            capsize=8, color=["steelblue", "darkorange"], alpha=0.85)
axes[1].set_ylabel("PR-AUC")
axes[1].set_title(f"PR-AUC across {len(SEEDS)} seeds (mean ± std)")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("variance_analysis.png", dpi=150)
plt.close()

print("\nSaved plots: pr_curve.png, confusion_matrix.png, "
      "probability_histogram.png, variance_analysis.png")

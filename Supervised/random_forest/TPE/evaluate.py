# -------------------------------------------------------------------------- USAGE ----------------------------------------------------------------------------- #
#       python evaluate.py --medium <default/vusp> --features <softdrop/shape/substructures> --pt_label <ptmin_ptmax> [--threshold 0.5] [--inverter on|off]      #
# -------------------------------------------------------------------------- ----- ----------------------------------------------------------------------------- #

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest Classifier evaluation.")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="test", help="Dataset for model evaluation: train / val / test")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Medium of the data used in predictions (where predictions were generated from).")
    parser.add_argument("--features", type=str, choices=['softdrop', 'shape', 'substructures'], required=True, help="Feature set.")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range in the format ptmin_ptmax")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics that need a hard label.")
    parser.add_argument("--inverter", type=str, choices=['on', 'off'], default='off', help="If 'on', the model was trained on the opposite medium of the data.")
    parser.add_argument("--balance", type=str, choices=["on", "off"], default="on", help="use model trained on balanced dataset only in train if on")
    return parser.parse_args()

def invert_medium(medium: str) -> str:
    return "vusp" if medium == "default" else "default"

def bootstrap_metric(metric_func, y_true, y_prob, threshold=0.5, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), len(y_true))
        # need both classes to compute most metrics reliably
        if len(np.unique(y_true[indices])) < 2:
            continue

        if metric_func == roc_auc_score:
            score = metric_func(y_true[indices], y_prob[indices])
        else:
            y_pred = (y_prob[indices] >= threshold).astype(int)
            score = metric_func(y_true[indices], y_pred)

        scores.append(score)

    scores = np.array(scores)
    return scores.mean(), scores.std()

# ------------------------------ #
#              MAIN              #
# ------------------------------ #
args = parse_args()
mode = args.mode
data_medium = args.medium               # medium of the DATA used to produce predictions
feature_set = args.features
pt_label = args.pt_label
threshold = args.threshold
inverter_tag = args.inverter
use_balancing = args.balance == "on"
print(f"Balance flag: {args.balance} (applies only to TRAIN datasets)\n")

# deduce training medium used by the model that produced the predictions
train_medium = invert_medium(data_medium) if inverter_tag == 'on' else data_medium

print()
print('**********************************************')
print('           EVALUATION CALCULATION STARTED     ')
print('**********************************************\n')
print(f'Data medium: [{data_medium}] | Model trained on: [{train_medium}] | Dataset: [{mode.upper()}] | Inverter: [{inverter_tag}]')
print('#------------------------------------------------------#\n')


scenario = "inverted" if inverter_tag == "on" else "standard"


base_pred_path = f"/sampa/llimadas/ML_models/random_forest/predictions/{scenario}/{feature_set}/{pt_label}/"
pred_path = os.path.join(base_pred_path, f"{pt_label}_{mode}_predictions_{feature_set}_train-{train_medium}_data-{data_medium}{'_balanced' if use_balancing else ''}.parquet")

# directory to save results
results_dir = f"/sampa/llimadas/ML_models/random_forest/results/{scenario}/{feature_set}/{pt_label}"
figures_dir = os.path.join(results_dir, "figures")
if use_balancing:
    figures_dir = os.path.join(figures_dir, "balanced")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

if not os.path.exists(pred_path):
    print(f"[WARNING] Prediction file not found: {pred_path}")
    raise SystemExit(0)

# loading predictions
df = pd.read_parquet(pred_path, engine='pyarrow')
y_true = df["True"].values
y_pred = df["Pred"].values
y_proba = df["Prob"].values

# 
# Confusion Matrix
#
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix\n{feature_set} | {pt_label.replace('_', ' - ')} GeV")
cm_base = f"cm_{feature_set}_{pt_label}_train-{train_medium}_data-{data_medium}{'_balanced' if use_balancing else ''}"
plt.savefig(os.path.join(figures_dir, f"{cm_base}.png"))
plt.savefig(os.path.join(figures_dir, f"{cm_base}.pdf"))
plt.close()

# --- #

cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
disp_norm.plot(cmap="Blues", values_format=".2f")
plt.title(f"Confusion Matrix (Normalized)\n{feature_set} | {pt_label.replace('_', ' - ')} GeV")
cm_base_norm = f"cm_norm_{feature_set}_{pt_label}_train-{train_medium}_data-{data_medium}{'_balanced' if use_balancing else ''}"
plt.savefig(os.path.join(figures_dir, f"{cm_base_norm}.png"))
plt.savefig(os.path.join(figures_dir, f"{cm_base_norm}.pdf"))
plt.close()

# 
# ROC Curve

fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve\n{feature_set} | {pt_label.replace('_', ' - ')} GeV")
plt.legend(loc="lower right")
roc_base = f"roc_{feature_set}_{pt_label}_train-{train_medium}_data-{data_medium}{'_balanced' if use_balancing else ''}"
plt.savefig(os.path.join(figures_dir, f"{roc_base}.png"))
plt.savefig(os.path.join(figures_dir, f"{roc_base}.pdf"))
plt.close()

# 
# Metrics with Bootstrap
# 
metrics = {
    "Accuracy": accuracy_score,
    "AUC": roc_auc_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1-Score": f1_score,
}

results = {}
for name, func in metrics.items():
    mean, std = bootstrap_metric(func, y_true, y_proba, threshold=threshold)
    results[name] = (mean, std)
    print(f"[{feature_set} | {pt_label}] -> {name}: {mean:.4f} ± {std:.4f}")

# Save metrics 
metrics_filename = f"metrics_{mode}_train-{train_medium}_data-{data_medium}{'_balanced' if use_balancing else ''}.txt"
with open(os.path.join(results_dir, metrics_filename), "w") as f:
    for name, (mean, std) in results.items():
        f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")

print("EVALUATIONS COMPLETED.")


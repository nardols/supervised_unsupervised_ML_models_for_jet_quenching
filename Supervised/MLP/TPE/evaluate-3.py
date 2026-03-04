import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of MLP predictions for jet classification.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Data medium: default or vusp")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], required=True, help="Mode: train / val / test")
    parser.add_argument("--pt_label", type=str, required=True, help="String like 'pTmin_pTmax'")
    parser.add_argument("--features", type=str, choices=['softdrop', 'shape', 'substructures'], required=True, help="Feature set")
    parser.add_argument("--threshold", type=float, required=False, default=0.5, help="Classification threshold")
    parser.add_argument("--inverter", type=str, choices=['on', 'off'], default='off', help="Enable inverse classification")
    parser.add_argument("--balance", type=str, choices=["on", "off"], default="on", help="use predictions on balanced dataset only in train if on")
    return parser.parse_args()

def invert_medium(medium: str) -> str:
    return "vusp" if medium == "default" else "default"

args = parse_args()
data_medium = args.medium
pt_label = args.pt_label
features = args.features
mode = args.mode
threshold = args.threshold
inverter_tag = args.inverter
use_balancing = args.balance == "on"

train_medium = invert_medium(data_medium) if inverter_tag == 'on' else data_medium
scenario = "inverted" if inverter_tag == "on" else "standard"

suffix = '_balanced' if use_balancing else ''
pred_dir = f"/sampa/llimadas/ML_models/MLP/predictions/{scenario}/{features}/{pt_label}/{mode}"
output_dir = f"/sampa/llimadas/ML_models/MLP/results/{scenario}/{features}/{pt_label}/{mode}{suffix}"
os.makedirs(output_dir, exist_ok=True)


pred_path = os.path.join(pred_dir, f"predictions_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.npy")
label_path = os.path.join(pred_dir, f"labels_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.npy")

y_true = np.load(label_path)
predictions = np.load(pred_path)
probs = predictions.flatten()

def bootstrap_metric(metric_func, y_true, y_prob, threshold=0.5, n_bootstrap=1000):
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_true), len(y_true))
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

metrics = {
    "Accuracy": accuracy_score,
    "AUC": roc_auc_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1-Score": f1_score,
}

results = {}
for name, func in metrics.items():
    mean, std = bootstrap_metric(func, y_true, probs, threshold)
    results[name] = (mean, std)
    print(f"{name}: {mean:.4f} ± {std:.4f}")

with open(os.path.join(output_dir, f"metrics_MLP_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.txt"), "w") as f:
    for name, (mean, std) in results.items():
        f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")

print("\nMetrics file saved.")

pt_min = pt_label.split('_')[0]
pt_max = pt_label.split('_')[-1]

fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = results["AUC"][0]

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - Jewel model (train: {train_medium} | data: {data_medium})\nfeatures: {features} in [{pt_min}, {pt_max}] GeV')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, f"roc_curve_MLP_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.pdf"))
plt.savefig(os.path.join(output_dir, f"roc_curve_MLP_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.png"))
plt.close()

print("ROC Curve saved.")

y_pred = (probs >= threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Quenched", "Quenched"])

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title(f'Confusion Matrix - Jewel model (train: {train_medium} | data: {data_medium})\nfeatures: {features} in [{pt_min}, {pt_max}] GeV')
plt.savefig(os.path.join(output_dir, f'confusion_matrix_MLP_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.pdf'))
plt.savefig(os.path.join(output_dir, f'confusion_matrix_MLP_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.png'))
plt.close()

print("Confusion Matrix saved.")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm / cm.sum(axis=1, keepdims=True),
    display_labels=["Not Quenched", "Quenched"]
)

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title(f'Normalized Confusion Matrix - Jewel model (train: {train_medium} | data: {data_medium})\nfeatures: {features} in [{pt_min}, {pt_max}] GeV')
plt.savefig(os.path.join(output_dir, f'confusion_matrix_MLP_{pt_label}_train-{train_medium}_data-{data_medium}_normalized{suffix}.pdf'))
plt.savefig(os.path.join(output_dir, f'confusion_matrix_MLP_{pt_label}_train-{train_medium}_data-{data_medium}_normalized{suffix}.png'))
plt.close()

print("Normalized Confusion Matrix saved.")
# -------------------------------------------------------------------------- USAGE ----------------------------------------------------------------------------- #
#  python evaluate.py --medium <default/vusp> --pt_label <ptmin_ptmax> --mode <train/val/test> [--threshold 0.5] [--inverter on|off] [--scaler on|off]           #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------- #

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="evaluating transformer predictions")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="medium of the dataset")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], default="off", help="scaler usage: on/off")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], required=True, help="dataset split")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range string in format 'ptmin_ptmax'")
    parser.add_argument("--inverter", type=str, choices=["on", "off"], default="off", help="using inverted training")
    parser.add_argument("--threshold", type=float, default=0.5, help="classification threshold")
    return parser.parse_args()


def invert_medium(m: str) -> str:
    return "vusp" if m == "default" else "default"


def bootstrap_metric(metric_func, y_true, y_prob, threshold=0.5, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
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


# ---------------- MAIN ---------------- #
def main():
    args = parse_args()

    pt_label = args.pt_label
    data_medium = args.medium
    mode = args.mode
    scaler = args.scaler
    threshold = args.threshold
    inverter_tag = args.inverter

    train_medium = invert_medium(data_medium) if inverter_tag == "on" else data_medium
    scenario = "inverted" if inverter_tag == "on" else "standard"

    print()
    print("**********************************************")
    print("       TRANSFORMER PREDICTION EVALUATION      ")
    print("**********************************************\n")
    print(f"dataset: [{mode.upper()}] | data medium: [{data_medium}] | "
          f"model trained on: [{train_medium}] | inverter: [{inverter_tag}] | "
          f"scaler: [{scaler}] | threshold: [{threshold}]\n")

    # setting directories
    pred_dir = f"/eos/user/l/llimadas/ML_models/Transformers/predictions/{scenario}/{pt_label}"
    results_dir = f"/eos/user/l/llimadas/ML_models/Transformers/results/{scenario}/{pt_label}/{mode}"
    figs_dir = os.path.join(results_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # prediction file paths
    pred_file = f"{pt_label}_{mode}_predictions_transformer_train-{train_medium}_data-{data_medium}_scaler-{scaler}.npy"
    label_file = f"{pt_label}_{mode}_labels_transformer_train-{train_medium}_data-{data_medium}_scaler-{scaler}.npy"
    pred_path = os.path.join(pred_dir, pred_file)
    label_path = os.path.join(pred_dir, label_file)

    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        print(f"[ERROR] prediction files not found for {pt_label}")
        return

    # loading predictions
    y_true = np.load(label_path)
    y_proba = np.load(pred_path)

    # -------------------------------- #
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])  # class 1 probs
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(f"ROC curve - {pt_label} ({mode})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(figs_dir, f"roc_curve_transformer_{pt_label}_{mode}_scaler-{scaler}.pdf"))
    plt.savefig(os.path.join(figs_dir, f"roc_curve_transformer_{pt_label}_{mode}_scaler-{scaler}.png"))
    plt.close()

    # -------------------------------- #
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Quenched", "Quenched"])
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"Confusion Matrix - {pt_label} ({mode})")
    plt.savefig(os.path.join(figs_dir, f"confusion_matrix_transformer_{pt_label}_{mode}_scaler-{scaler}.pdf"))
    plt.savefig(os.path.join(figs_dir, f"confusion_matrix_transformer_{pt_label}_{mode}_scaler-{scaler}.png"))
    plt.close()

    # plotting normalized confusion matrix
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Not Quenched", "Quenched"])
    disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title(f"Normalized Confusion Matrix - {pt_label} ({mode})")
    plt.savefig(os.path.join(figs_dir, f"confusion_matrix_normalized_transformer_{pt_label}_{mode}_scaler-{scaler}.pdf"))
    plt.savefig(os.path.join(figs_dir, f"confusion_matrix_normalized_transformer_{pt_label}_{mode}_scaler-{scaler}.png"))
    plt.close()

    # -------------------------------- #
    metrics = {
        "Accuracy": accuracy_score,
        "AUC": roc_auc_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1-Score": f1_score,
    }

    results = {}
    for name, func in metrics.items():
        mean, std = bootstrap_metric(func, y_true, y_proba[:, 1], threshold=threshold)
        results[name] = (mean, std)
        print(f"[{pt_label}] {name}: {mean:.4f} ± {std:.4f}")

    # saving metrics
    metrics_file = f"metrics_transformer_{pt_label}_{mode}_train-{train_medium}_data-{data_medium}_scaler-{scaler}.txt"
    with open(os.path.join(results_dir, metrics_file), "w") as f:
        f.write(f"fixed threshold: {threshold:.3f}\n")
        for name, (mean, std) in results.items():
            f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")

    # saving confusion matrix as text
    with open(os.path.join(results_dir, f"confusion_matrix_transformer_{pt_label}_{mode}_scaler-{scaler}.txt"), "w") as f:
        f.write(np.array2string(cm))

    print(f"\nTransformer evaluation completed for {pt_label}, results saved to {results_dir}")


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------- USAGE ----------------------------------------------------------------------------- #
# python evaluate.py --medium <default/vusp> --pt_label <ptmin_ptmax> [--mode train/val/test] [--threshold 0.5] [--inverter on|off] [--scaler on|off]
# -------------------------------------------------------------------------- ----- ----------------------------------------------------------------------------- #

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
import argparse


# ---------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="evaluating predictions from LSTM+Attention.")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="test", help="dataset split: train / val / test")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="medium of the dataset")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range in the format ptmin_ptmax")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], default="off", help="using scaled sequences: on / off")
    parser.add_argument("--inverter", type=str, choices=["on", "off"], default="off", help="using a model trained on the opposite medium")
    parser.add_argument("--threshold", type=float, default=0.5, help="fixed threshold for classification")
    return parser.parse_args()


# ---------------- #
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


# ---------------- #
# main script
# ---------------- #
def main():
    args = parse_args()

    mode = args.mode
    data_medium = args.medium
    pt_label = args.pt_label
    scaler_tag = args.scaler
    inverter_tag = args.inverter
    threshold = args.threshold

    train_medium = invert_medium(data_medium) if inverter_tag == "on" else data_medium
    scenario = "inverted" if inverter_tag == "on" else "standard"

    print('**********************************************')
    print('     LSTM + ATTENTION PREDICTION EVALUATION   ')
    print('**********************************************\n')
    print(f"dataset: [{mode.upper()}] | data medium: [{data_medium}] | "
          f"model trained on: [{train_medium}] | inverter: [{inverter_tag}] | "
          f"scaler: [{scaler_tag}] | threshold: [{threshold}]\n")

    # setting paths
    pred_dir = f"/eos/user/l/llimadas/ML_models/LSTM+Att/predictions/{scenario}/{pt_label}"
    results_dir = f"/eos/user/l/llimadas/ML_models/LSTM+Att/results/{scenario}/{pt_label}/{mode}"
    figs_dir = os.path.join(results_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # loading predictions
    pred_file = f"{pt_label}_{mode}_predictions_LSTM-att_train-{train_medium}_data-{data_medium}_scaler-{scaler_tag}.npy"
    label_file = f"labels_{pt_label}_{mode}_train-{train_medium}_data-{data_medium}_scaler-{scaler_tag}.npy"
    pred_path = os.path.join(pred_dir, pred_file)
    label_path = os.path.join(pred_dir, label_file)

    if not os.path.exists(pred_path) or not os.path.exists(label_path):
        print(f"error! prediction files not found: {pred_path} or {label_path}")
        sys.exit(1)
        return

    y_true = np.load(label_path)
    predictions = np.load(pred_path)
    probs = predictions[:, 1]  # taking probability of class 1 (quenched)

    min_len = min(len(y_true), len(probs))
    y_true = y_true[:min_len]
    probs = probs[:min_len]

    # plotting roc curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(f'ROC curve - {pt_label} ({mode})')
    plt.legend(loc="lower right")
    roc_base = f"roc_LSTM-att_{pt_label}_{mode}_train-{train_medium}_data-{data_medium}_scaler-{scaler_tag}"
    plt.savefig(os.path.join(figs_dir, f"{roc_base}.pdf"))
    plt.savefig(os.path.join(figs_dir, f"{roc_base}.png"))
    plt.close()

    # plotting confusion matrix
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Quenched", "Quenched"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix - {pt_label} ({mode})')
    cm_base = f"cm_LSTM-att_{pt_label}_{mode}_train-{train_medium}_data-{data_medium}_scaler-{scaler_tag}"
    plt.savefig(os.path.join(figs_dir, f"{cm_base}.pdf"))
    plt.savefig(os.path.join(figs_dir, f"{cm_base}.png"))
    plt.close()

    # plotting normalized confusion matrix
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Not Quenched", "Quenched"])
    disp_norm.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.title(f'Normalized Confusion Matrix - {pt_label} ({mode})')
    cm_norm_base = f"cm_norm_LSTM-att_{pt_label}_{mode}_train-{train_medium}_data-{data_medium}_scaler-{scaler_tag}"
    plt.savefig(os.path.join(figs_dir, f"{cm_norm_base}.pdf"))
    plt.savefig(os.path.join(figs_dir, f"{cm_norm_base}.png"))
    plt.close()

    metrics = {
        "Accuracy": accuracy_score,
        "AUC": roc_auc_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1-Score": f1_score,
    }

    results = {}
    for name, func in metrics.items():
        mean, std = bootstrap_metric(func, y_true, probs, threshold=threshold)
        results[name] = (mean, std)
        print(f"[{pt_label}] {name}: {mean:.4f} ± {std:.4f}")

    metrics_filename = f"metrics_{mode}_LSTM-att_train-{train_medium}_data-{data_medium}_scaler-{scaler_tag}.txt"
    with open(os.path.join(results_dir, metrics_filename), "w") as f:
        f.write(f"fixed threshold: {threshold:.3f}\n")
        for name, (mean, std) in results.items():
            f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")

    print(f"\nLSTM+Attention Evaluation completed! Results saved to {results_dir}")


if __name__ == "__main__":
    main()

import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model.MLP import MLP

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction with MLP for jet classification.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Data medium: default or vusp")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], required=True, help="Mode: train / val / test")
    parser.add_argument("--features", type=str, choices=['softdrop', 'shape', 'substructures'], required=True, help="Feature set")
    parser.add_argument("--pt_label", type=str, required=True, help="String like 'pTmin_pTmax'")
    parser.add_argument("--inverter", type=str, choices=['on', 'off'], default='off', help="Enable inverse classification")
    parser.add_argument("--balance", type=str, choices=["on", "off"], default="on", help="use model trained on balanced dataset only in train if on")
    return parser.parse_args()

def invert_medium(medium: str) -> str:
    return "vusp" if medium == "default" else "default"

@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        predictions.append(probs.cpu().numpy())
        true_labels.append(batch_y.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(true_labels)

def main():
    args = parse_args()

    pt_interval = args.pt_label
    data_medium = args.medium
    mode = args.mode
    feature_name = args.features
    use_balancing = args.balance == "on"

    train_medium = invert_medium(data_medium) if args.inverter == 'on' else data_medium
    scenario = "inverted" if args.inverter == "on" else "standard"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for pt_label in [pt_interval]:
        print(f"\nStarting prediction for pT: {pt_label}")

        suffix = '_balanced' if use_balancing else ''
        model_path  = f"/sampa/llimadas/ML_models/MLP/models/{train_medium}/{feature_name}/{pt_label}/best_model_MLP_{pt_label}{suffix}.pth"
        params_path = f"/sampa/llimadas/ML_models/MLP/models/{train_medium}/{feature_name}/{pt_label}/best_params_MLP_{pt_label}{suffix}.json"
        base_dir = f"/sampa/llimadas/nonseq_pre-processor/{data_medium}/{feature_name}/{pt_label}/"
        if use_balancing and mode == "train":
            dataset_path = os.path.join(base_dir, "balanced", f"{pt_label}_train_balanced.parquet")
        else:
            dataset_path = os.path.join(base_dir, f"{pt_label}_{mode}.parquet")


        # outputs
        out_dir = f"/sampa/llimadas/ML_models/MLP/predictions/{scenario}/{feature_name}/{pt_label}/{mode}"
        os.makedirs(out_dir, exist_ok=True)
        pred_file = f"predictions_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.npy"
        label_file = f"labels_{pt_label}_train-{train_medium}_data-{data_medium}{suffix}.npy"


        df = pd.read_parquet(dataset_path)
        X = df.drop(columns=['Type'])
        y = df['Type'].astype(np.float32)
        input_size = len(X.columns)

        with open(params_path, 'r') as f:
            best_params = json.load(f)
        hidden_sizes = [int(best_params['hidden_size0']), int(best_params['hidden_size1'])]
        dropout = float(best_params['dropout'])
        batch_size = int(best_params['num_batch'])

        dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = MLP(input_dim=input_size, hidden_size=hidden_sizes, dropout=dropout).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        preds, labels = predict(model, dataloader, device)

        np.save(os.path.join(out_dir, pred_file), preds)
        np.save(os.path.join(out_dir, label_file), labels)

        print(f"Saved for {pt_label}:\n -> {pred_file}\n -> {label_file}")

if __name__ == "__main__":
    main()

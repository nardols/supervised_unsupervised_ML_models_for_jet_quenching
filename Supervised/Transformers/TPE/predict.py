import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from model.JetTransformer import JetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction with Transformer for jet classification.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Medium of the dataset (used for evaluation).")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], default="off", help="Scaler usage: on/off (must match training).")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], required=True, help="Dataset split: train / val / test.")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range string in format 'ptmin_ptmax'.")
    parser.add_argument("--inverter", type=str, choices=["on", "off"], default="off", help="If 'on', load model trained on opposite medium.")
    return parser.parse_args()


def invert_medium(m: str) -> str:
    return "vusp" if m == "default" else "default"


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    for batch_x, batch_y, lengths in dataloader:
        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)

        max_len = batch_x.size(1)
        mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
        attention_mask = mask.unsqueeze(1).unsqueeze(2)

        logits = model(batch_x, mask=attention_mask)         # [batch, 2]
        probs = torch.softmax(logits, dim=1)                 # convert to probabilities

        predictions.append(probs.cpu().numpy())
        true_labels.append(batch_y.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(true_labels)


def load_data(data_path):
    df = pd.read_parquet(data_path)
    df = df[df["x_t"].map(len) > 0].reset_index(drop=True)
    sequences = [torch.tensor(np.stack(x), dtype=torch.float32) for x in df["x_t"] if len(x) > 0]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    X = pad_sequence(sequences, batch_first=True)
    y = torch.tensor(df["Type"].values, dtype=torch.long)

    return X, y, lengths

def main():
    args = parse_args()

    pt_label = args.pt_label
    data_medium = args.medium
    scenario = "inverted" if args.inverter == "on" else "standard"
    train_medium = invert_medium(data_medium) if args.inverter == "on" else data_medium

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n**********************************************")
    print("       TRANSFORMER PREDICTION CALCULATION     ")
    print("**********************************************\n")
    print(f"Dataset: [{args.mode.upper()}] | Data medium: [{data_medium}] | "
          f"Model trained on: [{train_medium}] | Inverter: [{args.inverter}] | Scaler: [{args.scaler}]\n")

    model_dir = f"/eos/user/l/llimadas/ML_models/Transformers/trained_models/{train_medium}/{pt_label}"
    model_path = os.path.join(model_dir, f"best_model_transformer_{pt_label}_scaler-{args.scaler}.pth")
    params_path = os.path.join(model_dir, f"best_params_transformer_{pt_label}.json")

    # output dir
    out_dir = f"/eos/user/l/llimadas/ML_models/Transformers/predictions/{scenario}/{pt_label}"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    if not os.path.exists(params_path):
        print(f"Params not found: {params_path}")
        return

    with open(params_path, "r") as f:
        best_params = json.load(f)

    # loading data
    data_path = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{data_medium}/{args.mode}/{pt_label}/{pt_label}_{args.mode}_scaler-{args.scaler}.parquet"
    X, y, len = load_data(data_path)

    # loading data from the training and val medium (to instance the model) 
    # ---> to simplify this, save the max_seq_lengths in the json file (i will not do it here because it would require training all the Transformers models again)
    data_train_path = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{train_medium}/train/{pt_label}/{pt_label}_train_scaler-{args.scaler}.parquet"
    _, _, lengths_train = load_data(data_train_path)
    data_val_path = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{train_medium}/val/{pt_label}/{pt_label}_val_scaler-{args.scaler}.parquet"
    _, _, lengths_val = load_data(data_val_path)
    max_seq_length = max(lengths_train.max().item(), lengths_val.max().item())

    # truncate X and len
    if X.size(1) > max_seq_length:
        X = X[:, :max_seq_length, :]
        len = torch.clamp(len, max=max_seq_length)
    dataset = TensorDataset(X, y, len)
    dataloader = DataLoader(dataset, batch_size=int(best_params["batch_size"]), shuffle=False)

    # loading model
    model = JetTransformer(
        input_dim=4,
        model_dim=int(best_params["model_dim"]),
        num_heads=int(best_params["num_heads"]),
        num_layers=int(best_params["num_layers"]),
        ff_dim=int(best_params["ff_dim"]),
        num_classes=2,
        dropout=best_params["dropout"],
        task="classification",
        max_seq_length=max_seq_length
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # running prediction
    preds, labels = predict(model, dataloader, device)

    # saving results
    np.save(os.path.join(out_dir, f"{pt_label}_{args.mode}_predictions_transformer_train-{train_medium}_data-{data_medium}_scaler-{args.scaler}.npy"), preds)
    np.save(os.path.join(out_dir, f"{pt_label}_{args.mode}_labels_transformer_train-{train_medium}_data-{data_medium}_scaler-{args.scaler}.npy"), labels)

    print(f"Transformers results saved to {out_dir}")


if __name__ == "__main__":
    main()

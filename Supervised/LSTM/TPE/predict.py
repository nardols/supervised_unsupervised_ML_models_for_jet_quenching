# -------------------------------------------------------------------------- USAGE ----------------------------------------------------------------------------- #
#              python predict.py --mode <train/val/test> --medium <default/vusp> --pt_label <ptmin_ptmax> [--scaler on|off] [--inverter on|off]                  #
# -------------------------------------------------------------------------- ----- ----------------------------------------------------------------------------- #

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
from model.LSTM import LSTM_FC 


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction for jet classification using LSTM.")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="test", help="Dataset split to predict: train / val / test")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Medium of the dataset used for prediction.")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range in the format ptmin_ptmax.")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], default="off", help="Use scaled sequences: on / off")
    parser.add_argument("--inverter", type=str, choices=["on", "off"], default="off", help="If 'on', use a model trained on the opposite medium.")
    return parser.parse_args()

def invert_medium(m: str) -> str:
    return "vusp" if m == "default" else "default"

@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    probs_all, labels_all, lengths_all = [], [], []
    for batch_x, batch_y, lengths in dataloader:
        batch_x, lengths = batch_x.to(device), lengths.to(device)
        logits = model(batch_x, lengths)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().ravel()
        probs_all.append(probs)
        labels_all.append(batch_y.numpy().ravel())
        lengths_all.append(lengths.cpu().numpy().ravel())
    return (
        np.concatenate(probs_all),
        np.concatenate(labels_all),
        np.concatenate(lengths_all)
    )

def main():
    
    args = parse_args()
    mode = args.mode
    data_medium = args.medium
    pt_label = args.pt_label
    scaler_tag = args.scaler
    inverter_tag = args.inverter
    
    train_medium = invert_medium(data_medium) if inverter_tag == "on" else data_medium
    scenario = "inverted" if inverter_tag == "on" else "standard"

    
    print('**********************************************')
    print('           LSTM PREDICTION CALCULATION        ')
    print('**********************************************\n')

    
    print(f"Dataset: [{mode.upper()}] | Data medium: [{data_medium}] | Model trained on: [{train_medium}] | Inverter: [{inverter_tag}] | Scaler: [{scaler_tag}]\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading trained model
    model_dir = f"/eos/user/l/llimadas/ML_models/LSTM/trained_models/{train_medium}/{pt_label}"
    model_path = os.path.join(model_dir, f"best_model_LSTM_{pt_label}_scaler-{scaler_tag}.pth")
    params_path = os.path.join(model_dir, f"best_params_LSTM_{pt_label}.json")
    
    data_path = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{data_medium}/{mode}/{pt_label}/{pt_label}_{mode}_scaler-{scaler_tag}.parquet"
    output_dir = f"/eos/user/l/llimadas/ML_models/LSTM/predictions/{scenario}/{pt_label}" 
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"{pt_label}_{mode}_predictions_LSTM_train-{train_medium}_data-{data_medium}_scaler-{scaler_tag}.parquet"
    output_path = os.path.join(output_dir, output_filename)

    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}"); sys.exit(1)
    if not os.path.exists(params_path):
        print(f"Params JSON not found: {params_path}"); sys.exit(1)
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}"); sys.exit(1)

    
    df = pd.read_parquet(data_path)
    df = df[df['x_t'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    
    sequences = [torch.tensor(np.stack(x), dtype=torch.float32) for x in df["x_t"]]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    X = pad_sequence(sequences, batch_first=True)
    y = torch.tensor(df["Type"].values, dtype=torch.long)
    
    dataset = TensorDataset(X, y, lengths)
    
    with open(params_path, "r") as f:
        best_params = json.load(f)
        
    try:
        batch_size = int(best_params["num_batch"])
        hidden_sizes = [int(best_params["hidden_size0"]), int(best_params["hidden_size1"])]
        num_layers = int(best_params["num_layers"])
    except KeyError as e:
        print(f"Missing key in params JSON {params_path}: {e}")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = LSTM_FC(input_size=4, hidden_size=hidden_sizes, num_layers=num_layers, batch_size=batch_size, device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    y_proba, y_true, seq_lengths = predict(model, dataloader, device)
    y_pred = (y_proba >= 0.5).astype(int)
    
    result_df = pd.DataFrame({"True": y_true, "Prob": y_proba, "Length": seq_lengths})
    result_df.to_parquet(output_path, index=False); print(f"Results saved to: {output_path}")
    print(classification_report(y_true, y_pred, digits=3)); print("\nPREDICTIONS COMPLETED.")

if __name__ == "__main__":
    main()


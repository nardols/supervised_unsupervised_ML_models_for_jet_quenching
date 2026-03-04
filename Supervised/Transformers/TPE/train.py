import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from model.JetTransformer import JetTransformer
import torch.nn as nn
import random

# setting seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Transformer model for jet classification.")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], required=True, help="Apply scaler to sequences: on / off.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Medium type used in Jewel: default/vusp.")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range string in format 'ptmin_ptmax'.")
    return parser.parse_args()

args = parse_args()
scaler_tag = args.scaler
medium = args.medium
pt_interval = args.pt_label

pt_min = float(pt_interval.split('_')[0])
pt_max = float(pt_interval.split('_')[-1])
pt_ranges = {pt_interval: (pt_min, pt_max)}

# -------------------------------
# hyperparameter search space
space = hp.choice('hyper_parameters', [
    {
        'model_dim': hp.choice('model_dim', [32, 64, 128]),
        'num_heads': hp.choice('num_heads', [2, 4, 8]),
        'num_layers': hp.quniform('num_layers', 2, 6, 1),
        'dropout': hp.uniform('dropout', 0.05, 0.5),
        'ff_dim': hp.choice('ff_dim', [128, 256, 512]),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(5e-3)),
        'decay': hp.uniform('decay', 0.8, 0.99),
        'num_epochs': hp.quniform('num_epochs', 20, 40, 5),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256])
    }
])

# -------------------------------
def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

# -------------------------------
def train_model(params, train_dataset, val_dataset, model_dir, pt_label, scaler_tag, max_seq_length):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = 4
    num_classes = 2
    model_dim = int(params['model_dim'])
    num_heads = int(params['num_heads'])
    num_layers = int(params['num_layers'])
    dropout = params['dropout']
    ff_dim = int(params['ff_dim'])
    lr = params['lr']
    decay = params['decay']
    batch_size = int(params['batch_size'])
    num_epochs = int(params['num_epochs'])

    model = JetTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout,
        task="classification",
        max_seq_length=max_seq_length
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch, lengths in train_loader:
            x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)

            max_len = x_batch.size(1)
            mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
            attention_mask = mask.unsqueeze(1).unsqueeze(2)

            output = model(x_batch, mask=attention_mask)  # shape [batch, num_classes]

            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, lengths in val_loader:
                x_val, y_val, lengths = x_val.to(device), y_val.to(device), lengths.to(device)

                max_len_val = x_val.size(1)
                mask_val = torch.arange(max_len_val, device=device)[None, :] < lengths[:, None]
                attention_mask_val = mask_val.unsqueeze(1).unsqueeze(2)

                val_output = model(x_val, mask=attention_mask_val)
                loss = criterion(val_output, y_val)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    return {
        'loss': best_loss,
        'status': STATUS_OK,
        'model': best_state,
        'params': params,
        'losses': {
            "train_losses": train_losses,
            "val_losses": val_losses
        }
    }

# -------------------------------
# main
# -------------------------------
if __name__ == "__main__":
    for pt_label in pt_ranges:
        # directories
        model_dir = f"/eos/user/l/llimadas/ML_models/Transformers/trained_models/{medium}/{pt_label}"
        dataset_dir = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{medium}/train/{pt_label}"
        validation_dir = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{medium}/val/{pt_label}"
        os.makedirs(model_dir, exist_ok=True)

        print(f"\nTraining transformer for pT range: {pt_label}")

        # loading train data
        file_path = os.path.join(dataset_dir, f"{pt_label}_train_scaler-{scaler_tag}.parquet")
        df = pd.read_parquet(file_path)
        sequences = [torch.tensor(np.stack(x), dtype=torch.float32) for x in df['x_t'] if len(x) > 0]
        lengths_train = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        y_train = torch.tensor(df.loc[df['x_t'].apply(len) > 0, 'Type'].values, dtype=torch.long)
        x_train = pad_sequence(sequences, batch_first=True)
        train_dataset = TensorDataset(x_train, y_train, lengths_train)

        # loading validation data
        file_val_path = os.path.join(validation_dir, f"{pt_label}_val_scaler-{scaler_tag}.parquet")
        df_val = pd.read_parquet(file_val_path)
        sequences_val = [torch.tensor(np.stack(x), dtype=torch.float32) for x in df_val['x_t'] if len(x) > 0]
        lengths_val = torch.tensor([len(seq) for seq in sequences_val], dtype=torch.long)
        y_val = torch.tensor(df_val.loc[df_val['x_t'].apply(len) > 0, 'Type'].values, dtype=torch.long)
        x_val = pad_sequence(sequences_val, batch_first=True)
        val_dataset = TensorDataset(x_val, y_val, lengths_val)

        max_seq_length = max(lengths_train.max().item(), lengths_val.max().item())

        # running hyperparameter optimization
        trials = Trials()
        best = fmin(
            fn=lambda params: train_model(params, train_dataset, val_dataset, model_dir, pt_label, scaler_tag, max_seq_length),
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials,
            rstate=np.random.default_rng(SEED)
        )

        # retrieving best results
        best_trial = min([t for t in trials.trials if t['result']['status'] == STATUS_OK], key=lambda t: t['result']['loss'])
        best_state = best_trial['result']['model']
        best_params = convert_to_serializable(best_trial['result']['params'])
        best_losses = best_trial['result']['losses']

        # saving model and params
        torch.save(best_state, os.path.join(model_dir, f"best_model_transformer_{pt_label}_scaler-{scaler_tag}.pth"))
        with open(os.path.join(model_dir, f"best_params_transformer_{pt_label}.json"), 'w') as f:
            json.dump(best_params, f, indent=4)
        with open(os.path.join(model_dir, f"losses_transformer_{pt_label}_scaler-{scaler_tag}.json"), 'w') as f:
            json.dump(best_losses, f, indent=4)

        print(f"Transformer model and hyperparameters saved for pT: {pt_label}")






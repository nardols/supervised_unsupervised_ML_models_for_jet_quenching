import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
import argparse
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from model.LSTM_Att import LSTM_Attention
import random
import torch.nn as nn

# --------------------- #

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Sequence preprocessing for jet classification (LSTM + Attention).")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], required=True, help="Apply scaling to sequences: on/off")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Medium type used in Jewel: default/vusp")
    parser.add_argument("--pt_label", type=str, required=True, help="String of type 'pTmin_pTmax'.")  
    return parser.parse_args()

args = parse_args()
scaler_tag = args.scaler
medium = args.medium
pt_interval = args.pt_label

# --------------------- #

pt_min = float(pt_interval.split('_')[0])
pt_max = float(pt_interval.split('_')[-1])
pt_ranges = {pt_interval: (pt_min, pt_max)}  # compatible with old code, but here we want to run with HTCondor

""" Example for multiple ranges:
pt_ranges = {
    '40_60': (40, 60),
    '60_80': (60, 80),
    '80_120': (80, 120),
    '120_200': (120, 200),
    '200_400': (200, 400),
    '80_250': (80, 250),
}
"""

# --------------------- #
# Hyperparameter space
# --------------------- #
space = hp.choice('hyper_parameters', [
    {
        'num_batch': hp.choice('num_batch', [64, 128, 256]),
        'num_epochs': hp.quniform('num_epochs', 40, 50, 5),
        'num_layers': hp.quniform('num_layers', 2, 4, 1),
        'dropout': hp.uniform('dropout', 0.1, 0.5),
        'hidden_size0': hp.quniform('hidden_size0', 8, 20, 2),
        'hidden_size1': hp.quniform('hidden_size1', 4, 8, 2),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.05),
        'decay_factor': hp.uniform('decay_factor', 0.9, 0.99),
        'loss_func': hp.choice('loss_func', ['mse', 'bce'])
    }
])

# --------------------- #
# Functions
# --------------------- #

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2) / torch.sum(weight)

def weighted_bce_loss(input, target, weight):
    return torch.nn.functional.binary_cross_entropy(input, target, weight, reduction='sum') / torch.sum(weight)

# --------------------- #

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

# --------------------- #

def train_model(params, train_dataset, val_dataset, model_dir, pt_label, scaler_tag):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = int(params['num_batch'])
    num_epochs = int(params['num_epochs'])
    num_layers = int(params['num_layers'])
    hidden_sizes = [int(params['hidden_size0']), int(params['hidden_size1'])]
    lr = params['learning_rate']
    decay = params['decay_factor']

    # define model
    model = LSTM_Attention(
        input_size=4,
        hidden_size=hidden_sizes,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    # CrossEntropyLoss
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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)  # keep as class indices (0/1)
            lengths = lengths.to(device)

            output = model(x_batch, lengths)  # (batch, 2) logits

            loss = criterion(output, y_batch)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val, lengths in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                lengths = lengths.to(device)

                val_output = model(X_val, lengths)
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

# --------------------- #
#         MAIN          #
# --------------------- #
if __name__ == "__main__":

    for pt_label in pt_ranges:

        # Directories
        model_dir = f"/eos/user/l/llimadas/ML_models/LSTM+Att/trained_models/{medium}/{pt_label}"  # to save the trained model
        dataset_dir = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{medium}/train/{pt_label}"  
        validation_dir = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{medium}/val/{pt_label}" 
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nTraining model for pT range: {pt_label}")

        ###################
        #  Preparing data #
        ###################

        # -- train data --#
        file_path = os.path.join(dataset_dir, f"{pt_label}_train_scaler-{scaler_tag}.parquet")
        df = pd.read_parquet(file_path)

        print("Minimum train sequence length:", min(len(x) for x in df['x_t']))

        sequences = [
            torch.tensor(np.stack(x), dtype=torch.float32) for x in df['x_t'] if len(x) > 0
        ]
        lengths_train = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        y_train = torch.tensor(df.loc[df['x_t'].apply(len) > 0, 'Type'].values, dtype=torch.long)
        X_train = pad_sequence(sequences, batch_first=True)
        train_dataset = TensorDataset(X_train, y_train, lengths_train)

        # -- validation data --#
        file_val_path = os.path.join(validation_dir, f"{pt_label}_val_scaler-{scaler_tag}.parquet")
        df_val = pd.read_parquet(file_val_path)

        sequences_val = [
            torch.tensor(np.stack(x), dtype=torch.float32) for x in df_val['x_t'] if len(x) > 0
        ]
        lengths_val = torch.tensor([len(seq) for seq in sequences_val], dtype=torch.long)
        y_val = torch.tensor(df_val.loc[df_val['x_t'].apply(len) > 0, 'Type'].values, dtype=torch.long)
        X_val = pad_sequence(sequences_val, batch_first=True)
        val_dataset = TensorDataset(X_val, y_val, lengths_val)
        
        #################

        trials = Trials()
        best = fmin(
            fn=lambda params: train_model(params, train_dataset, val_dataset, model_dir, pt_label, scaler_tag), 
            space=space, 
            algo=tpe.suggest, 
            max_evals=5, 
            trials=trials,
            rstate=np.random.default_rng(SEED)
        )

        best_trial = min([t for t in trials.trials if t['result']['status'] == STATUS_OK], key=lambda t: t['result']['loss'])
        best_state = best_trial['result']['model']
        best_params = convert_to_serializable(best_trial['result']['params'])
        best_losses = best_trial['result']['losses']

        # Save model
        torch.save(best_state, os.path.join(model_dir, f"best_model_LSTM-att_{pt_label}_scaler-{scaler_tag}.pth"))
        with open(os.path.join(model_dir, f"best_params_LSTM-att_{pt_label}.json"), 'w') as f:
            json.dump(best_params, f, indent=4)

        with open(os.path.join(model_dir, f"losses_LSTM-att_{pt_label}_scaler-{scaler_tag}.json"), 'w') as f:
            json.dump(best_losses, f, indent=4)

        print(f"\n[OK] Model and hyperparameters saved for pT range: {pt_label}")

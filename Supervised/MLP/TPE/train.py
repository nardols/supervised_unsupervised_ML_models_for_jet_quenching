import os
import numpy as np
import pandas as pd
import argparse
from model.MLP import MLP
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import json
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Training of a simple MLP for jet classification.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Type of medium used in Jewel: default/vusp")
    parser.add_argument("--features", type=str, choices=['softdrop', 'shape', 'substructures'], required=True, help="Set of substructures")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range in format ptmin_ptmax")
    parser.add_argument("--balance", type=str, choices=["on", "off"], default="on", help="use balanced dataset only in train if on")
    return parser.parse_args()

# arguments
args = parse_args()
medium = args.medium
feature_set = args.features
pt_label = args.pt_label
use_balancing = args.balance == "on"

# pT ranges
pt_ranges = {
    '40_60': (40, 60),
    '60_80': (60, 80),
    '80_120': (80, 120),
    '120_200': (120, 200),
    '200_400': (200, 400),
    '80_250': (80, 250),
}
pt_ranges = {pt_label: pt_ranges[pt_label]}

# hyperparameter space
space = {
    'hidden_size0': hp.choice('hidden_size0', [64, 32, 16, 12]),
    'hidden_size1': hp.choice('hidden_size1', [16, 12, 10, 8]),
    'num_batch': hp.quniform('num_batch', 32, 512, 32),
    'learning_rate': hp.uniform('learning_rate', 1e-4, 5e-3),
    'decay_factor': hp.uniform('decay_factor', 0.9, 0.99),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    'num_epochs': hp.quniform('num_epochs', 30, 100, 10)
}

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

def train_model(params, input_size, train_dataset, val_dataset, model_dir, pt_label):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = int(params['num_batch'])
    num_epochs = int(params['num_epochs'])
    hidden_sizes = [int(params['hidden_size0']), int(params['hidden_size1'])]
    lr = float(params['learning_rate'])
    decay = float(params['decay_factor'])
    dropout = float(params['dropout'])

    model = MLP(input_dim=input_size, hidden_size=hidden_sizes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # computing class weights for cross entropy
    # y_all = train_dataset.tensors[1].cpu().numpy().astype(int)
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_all)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_state = None
    
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.float().to(device)  

            output = model(x_batch) 
            loss = criterion(output, y_batch)

            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.float().to(device)

                val_output = model(X_val)
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

print('**********************************************')
print('    STARTING TRAINING OF THE SIMPLE MLP       ')
print('**********************************************\n')
print(f'Medium: [{medium}]\n')

if __name__ == "__main__":
    for pt_label in pt_ranges:
        model_dir = f"/sampa/llimadas/ML_models/MLP/models/{medium}/{feature_set}/{pt_label}"
        dataset_dir =  f"/sampa/llimadas/nonseq_pre-processor/{medium}/{feature_set}/{pt_label}"
        validation_dir =  f"/sampa/llimadas/nonseq_pre-processor/{medium}/{feature_set}/{pt_label}"
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nTraining model for pT range: {pt_label}")

        # train data
        if use_balancing:
            file_path = os.path.join(dataset_dir, "balanced", f"{pt_label}_train_balanced.parquet")
        else:
            file_path = os.path.join(dataset_dir, f"{pt_label}_train.parquet")

        df = pd.read_parquet(file_path)
        y_train = df['Type'].astype(int)
        X_train = df.drop(columns=['Type'])
        train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
        
        # val data
        file_val_path = os.path.join(validation_dir, f"{pt_label}_val.parquet")
        df_val = pd.read_parquet(file_val_path)
        y_val = df_val['Type'].astype(int)
        X_val = df_val.drop(columns=['Type'])
        val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.long))

        input_size = len(X_train.columns)

        trials = Trials()
        best = fmin(
            fn=lambda params: train_model(params, input_size, train_dataset, val_dataset, model_dir, pt_label), 
            space=space, 
            algo=tpe.suggest, 
            max_evals=5, 
            trials=trials,
            rstate=np.random.default_rng(42)
        )

        best_trial = min([t for t in trials.trials if t['result']['status'] == STATUS_OK], key=lambda t: t['result']['loss'])
        best_state = best_trial['result']['model']
        best_params = convert_to_serializable(best_trial['result']['params'])
        best_losses = best_trial['result']['losses']
        
        torch.save(best_state, os.path.join(model_dir, f"best_model_MLP_{pt_label}{'_balanced' if use_balancing else ''}.pth"))
        with open(os.path.join(model_dir, f"best_params_MLP_{pt_label}{'_balanced' if use_balancing else ''}.json"), 'w') as f:
            json.dump(best_params, f, indent=4)
        with open(os.path.join(model_dir, f"losses_MLP_{pt_label}{'_balanced' if use_balancing else ''}.json"), 'w') as f:
            json.dump(best_losses, f, indent=4)

        print(f"\nMLP model and hyperparameters saved for pT: {pt_label}")

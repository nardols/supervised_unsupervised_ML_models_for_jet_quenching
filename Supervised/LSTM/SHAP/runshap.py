import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import pandas as pd 
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append("/eos/user/l/llimadas/ML_models/LSTM/TPE/model")
from LSTM import LSTM_FC
from captum.attr import GradientShap


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP analysis for jet classification with LSTM")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True)
    parser.add_argument("--scaler", type=str, choices=["on", "off"], default="off")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--pt_label", type=str, required=True)  
    return parser.parse_args()


def apply_shap(model, dataloader, device, num_samples, train_dataset):
    print("Starting SHAP attribution calculation...")
    model.eval()
    
    g_shap = GradientShap(model)
    rand_indices = np.random.choice(len(train_dataset), 100, replace=False)
    baseline_distribution = train_dataset.tensors[0][rand_indices].to(device)
    baseline_sequences = train_dataset.tensors[0][rand_indices] 
    
    all_feature_importances, all_timestep_importances, all_attributions = [], [], []
    samples_explained = 0

    for batch_x, batch_y, lengths in dataloader:
        if samples_explained >= num_samples:
            break

        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)
        bs = batch_x.size(0)
        max_len_batch = batch_x.size(1)
        
        if baseline_sequences.size(0) >= bs:
            selected_baselines = baseline_sequences[:bs]
        else:
            reps = int(np.ceil(bs / baseline_sequences.size(0)))
            selected_baselines = baseline_sequences.repeat(reps, 1, 1)[:bs]
        
        baseline_max_len = selected_baselines.size(1)
        
        if baseline_max_len > max_len_batch:
            baselines = selected_baselines[:, :max_len_batch, :].to(device)
        elif baseline_max_len < max_len_batch:
            padding = torch.zeros(
                bs, 
                max_len_batch - baseline_max_len, 
                selected_baselines.size(2)
            )
            baselines = torch.cat([selected_baselines, padding], dim=1).to(device)
        else:
            baselines = selected_baselines.to(device)

        outputs = model(batch_x, lengths)
        predicted_classes = torch.argmax(outputs, dim=1)

        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        attributions = g_shap.attribute(
            batch_x, 
            baselines=baselines, 
            n_samples=100, 
            additional_forward_args=(lengths,), 
            target=batch_y
        )
        model.eval()
        
        all_attributions.append(attributions.cpu())
        abs_attr = torch.abs(attributions)
        
        all_feature_importances.append(torch.sum(abs_attr, dim=1).cpu())
        all_timestep_importances.append(torch.sum(abs_attr, dim=2).cpu())
        
        samples_explained += bs

    global_feature_importance = torch.mean(torch.cat(all_feature_importances, dim=0), dim=0) 
    global_timestep_importance = torch.mean(torch.cat(all_timestep_importances, dim=0), dim=0)
    #evolution_importance = torch.mean(torch.cat(all_attributions, dim=0), dim=0)
    evolution_importance = abs_attr.mean(dim=0).cpu()   
    full_attributions = torch.cat(all_attributions, dim=0)

    return global_feature_importance, global_timestep_importance, evolution_importance, full_attributions


def load_data(dataset_path):
    df = pd.read_parquet(dataset_path)
    sequences = [torch.tensor(np.stack(x), dtype=torch.float32) for x in df['x_t'] if len(x) > 0]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    y = torch.tensor(df.loc[df['x_t'].apply(len) > 0, 'Type'].values, dtype=torch.long)
    X = pad_sequence(sequences, batch_first=True)
    return TensorDataset(X, y, lengths)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Processing pT: {args.pt_label}")
    
    base_path = f"/eos/user/l/llimadas/ML_models/LSTM/trained_models/{args.medium}/{args.pt_label}"
    model_path = f"{base_path}/best_model_LSTM_{args.pt_label}_scaler-{args.scaler}.pth"
    params_path = f"{base_path}/best_params_LSTM_{args.pt_label}.json"
    
    dataset_path = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{args.medium}/{args.mode}/{args.pt_label}/{args.pt_label}_{args.mode}_scaler-{args.scaler}.parquet"
    train_path = f"/eos/user/l/llimadas/seq_pre-processor/processed_data/results/{args.medium}/train/{args.pt_label}/{args.pt_label}_train_scaler-{args.scaler}.parquet"
    
    out_dir = f"/eos/user/l/llimadas/ML_models/LSTM/shap/results/{args.medium}/{args.pt_label}/{args.mode}"
    
    dataset = load_data(dataset_path)
    train_dataset = load_data(train_path)
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    hidden_sizes = [int(params['hidden_size0']), int(params['hidden_size1'])]
    num_layers = int(params['num_layers'])
    batch_size = int(params['num_batch'])
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = LSTM_FC(
        input_size=4,
        hidden_size=hidden_sizes,
        num_layers=num_layers,
        batch_size=batch_size,
        device=device
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    global_feat_imp, global_time_imp, evolution_imp, full_attr = apply_shap(model, dataloader, device, len(dataset), train_dataset)
    #apply_shap(model, dataloader, device, len(dataset), dataset)
    
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(f"{out_dir}/global_feature_importance_{args.pt_label}_scaler-{args.scaler}.npy", global_feat_imp.numpy())
    np.save(f"{out_dir}/global_timestep_importance_{args.pt_label}_scaler-{args.scaler}.npy", global_time_imp.numpy())
    np.save(f"{out_dir}/evolution_importance_{args.pt_label}_scaler-{args.scaler}.npy", evolution_imp.numpy())
    np.save(f"{out_dir}/per_sample_attributions_{args.pt_label}_scaler-{args.scaler}.npy", full_attr.numpy())
    
    print(f"SHAP attributions saved to: {out_dir}")


if __name__ == "__main__":
    main()
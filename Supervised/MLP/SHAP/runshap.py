import shap
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import argparse
import sys

sys.path.append("/eos/user/l/llimadas/ML_models/MLP/TPE")
from model.MLP import MLP


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP values for MLP jet quenching classification")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Type of medium in Jewel: default/vusp")
    return parser.parse_args()

# ---------------------------- #
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 22,
    'figure.titlesize': 28,
    'figure.dpi': 300,
    'font.size': 24
})
# ---------------------------- #

args = parse_args()
medium = args.medium
features = ['softdrop']
pt_ranges = ['40_60', '80_250', '200_400']


# ---------------------------- #
rename_dict = {
    'ievt': r'$i_{\mathrm{evt}}$',
    'ijet': r'$i_{\mathrm{jet}}$',
    'evwt': r'$w_{\mathrm{evt}}$',
    'pt': r'$p_{\mathrm{T}}$',
    'eta': r'$\eta$',
    'rapidity': r'$y$',
    'phi': r'$\phi$',
    'nconst': r'$N_{\mathrm{const}}$',
    'zg': r'$z_{g}$',
    'Rg': r'$R_{g}$',
    'kg': r'$k_g$',
    'nSD': r'$n_{\mathrm{SD}}$',
    'mass': r'$m$',
    'mz2': r'$\bar{r}_{SD}$',
    'mr': r'$\bar{r}^2_{SD}$',
    'mr2': r'$rz_{SD}$',
    'rz': r'$r^2z_{SD}$',
    'r2z': r'$\bar{z}^2_{SD}$',
    'ptd': r'$p_TD_{SD}$',
    'jetcharge03': r'$Q_{0.3}$',
    'jetcharge05': r'$Q_{0.5}$',
    'jetcharge07': r'$Q_{0.7}$',
    'jetcharge10': r'$Q_{1.0}$',
    'tau1': r'$\tau_{1}$',
    'tau2': r'$\tau_{2}$',
    'tau3': r'$\tau_{3}$',
    'tau4': r'$\tau_{4}$',
    'tau5': r'$\tau_{5}$',
    'tau2tau1': r'$\tau_{2, 1}$',
    'tau3tau2': r'$\tau_{3, 2}$',
    'kappa_TD': r'$\kappa_{\mathrm{TD}}$',
    'kappa_ktD': r'$\kappa_{\mathrm{ktD}}$',
    'kappa_zD': r'$\kappa_{\mathrm{zD}}$',
    'zg_TD': r'$z_{g,\mathrm{TD}}$',
    'zg_ktD': r'$z_{g,\mathrm{ktD}}$',
    'zg_zD': r'$z_{g,\mathrm{zD}}$',
    'deltaR_TD': r'$R_{g, \mathrm{TD}}$',
    'deltaR_ktD': r'$R_{g, \mathrm{ktD}}$',
    'deltaR_zD': r'$R_{g, \mathrm{zD}}$',
    
    'SD_pt': r'$p_{\mathrm{T}}^{\mathrm{SD}}$',
    'SD_eta': r'$\eta^{\mathrm{SD}}$',
    'SD_rapidity': r'$y_{\mathrm{SD}}$',
    'SD_phi': r'$\phi_{\mathrm{SD}}$',
    'SD_nconst': r'$n_{\mathrm{const}, \mathrm{SD}}$',
    'SD_mass': r'$m_{\mathrm{g}}$',  
    'SD_mz2': r'$ z^{2}, \mathrm{SD}$',
    'SD_mr': r'$\bar{r}_{SD}$',
    'SD_mr2': r'$\bar{r}^2_{SD}$',
    'SD_rz': r'$r z_{SD}$',
    'SD_r2z': r'$r^2z_{SD}$',
    'SD_ptd': r'$p_T D_{\mathrm{SD}}$',
    'SD_jetcharge03': r'$Q^{0.3}_{\mathrm{SD}}$',
    'SD_jetcharge05': r'$Q^{0.5}_{\mathrm{SD}}$',
    'SD_jetcharge07': r'$Q^{0.7}_{\mathrm{SD}}$',
    'SD_jetcharge10': r'$Q^{1.0}_{\mathrm{SD}}$',
    'SD_tau1': r'$\tau_{1}^{\mathrm{SD}}$',
    'SD_tau2': r'$\tau_{2}^{\mathrm{SD}}$',
    'SD_tau3': r'$\tau_{3}^{\mathrm{SD}}$',
    'SD_tau4': r'$\tau_{4}^{\mathrm{SD}}$',
    'SD_tau5': r'$\tau_{5}^{\mathrm{SD}}$',
    'SD_tau2tau1': r'$\tau_{2,1,\mathrm{SD}}$',
    'SD_tau3tau2': r'$\tau_{3,2, \mathrm{SD}}$'
}
# ---------------------------- #

for feature in features:
    for pt_label in pt_ranges:
        model_path = f"/eos/user/l/llimadas/ML_models/MLP/models/{medium}/{feature}/{pt_label}/best_model_MLP_{pt_label}.pth"
        data_path = f"/eos/user/l/llimadas/nonseq_pre-processor/{medium}/{feature}/{pt_label}/{pt_label}_test.parquet"
        params_path = f"/eos/user/l/llimadas/ML_models/MLP/models/{medium}/{feature}/{pt_label}/best_params_MLP_{pt_label}.json"

        df = pd.read_parquet(data_path)
        X = df.drop(columns=['Type'])
        X = X.rename(columns=rename_dict)
        y = df['Type'].values

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(params_path, 'r') as f:
            params = json.load(f)

        input_size = len(X.columns)
        hidden_size = [int(params['hidden_size0']), int(params['hidden_size1'])]
        dropout = params['dropout']

        model = MLP(input_size, hidden_size, dropout).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

        def model_predict(x_numpy):
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.sigmoid(logits) 
            return probs.cpu().numpy()
        
        background = X.sample(2000, random_state=42).values
        
        explainer = shap.Explainer(model_predict, background) 
        exp = explainer(X.values, max_evals=2072)
        exp.feature_names = list(X.columns)
        
        shap.plots.beeswarm(exp)
        save_dir = f"/eos/user/l/llimadas/ML_models/MLP/shap/figures/{medium}/{feature}/{pt_label}"
        os.makedirs(save_dir, exist_ok=True)
        plt.title(f"$p_T \in$ [{pt_label.replace('_', ', ')}] GeV", fontsize=16)
        plt.savefig(os.path.join(save_dir, f"plot_MLP_beeswarm_{medium}_{pt_label}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"plot_MLP_beeswarm_{medium}_{pt_label}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

        shap.plots.bar(exp)
        plt.title(f"$p_T \in$ [{pt_label.replace('_', ', ')}] GeV", fontsize=16)
        plt.savefig(os.path.join(save_dir, f"plot_MLP_bar_{medium}_{pt_label}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"plot_MLP_bar_{medium}_{pt_label}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

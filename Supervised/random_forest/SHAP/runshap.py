import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import argparse
from sklearn.ensemble import RandomForestClassifier
import json

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
    'SD_rapidity': r'$y^{\mathrm{SD}}$',
    'SD_phi': r'$\phi^{\mathrm{SD}}$',
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

def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest importance in jet quenching classification.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Medium type used in Jewel: default/vusp")
    return parser.parse_args()

args = parse_args()

# ---------------------------- #

def plot_polar_area_chart(features, importances, title, save_path):
    percentages = 100 * np.array(importances) / np.sum(importances)
    N = len(features)
    
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = percentages  
    width = 2 * np.pi / N 

    colors = plt.cm.BuPu(np.linspace(0.3, 0.85, N))

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    bars = ax.bar(theta, radii, width=width, bottom=0.0, color=colors, edgecolor='white')

    for i, (bar, label, pct) in enumerate(zip(bars, features, percentages)):
        angle = theta[i]
        r = radii[i] + 3  
        ax.text(angle, r, f"{label}\n{pct:.1f}\%", ha='center', va='center', fontsize=16)

    ax.set_title(title, va='bottom', fontsize=14)
    ax.set_yticklabels([])  
    ax.set_xticklabels([])  
    ax.grid(False)
    ax.spines['polar'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path + ".png")
    plt.savefig(save_path + ".pdf")
    plt.close()


# ---------------------------- #

medium = args.medium
base_model_path = f"/eos/user/l/llimadas/ML_models/random_forest/models/{medium}/"

feature_sets = {
    'softdrop': ['zg', 'Rg', 'kg', 'SD_mass'],
}

#pt_ranges = ['40_60', '60_80', '80_120', '120_200', '200_400', '80_250']
pt_ranges = ['40_60', '200_400', '80_250']

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,                
    "font.family": "serif",
    "font.serif": ["Computer Modern"],   
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 22,
    "figure.titlesize": 28,
    "figure.dpi": 300,
    "font.size": 24,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}", 
})

# ---------------------------- #
#             loop             #
# ---------------------------- #

for feature_name, features in feature_sets.items():
    importance_dict = {feat: [] for feat in features}
    pt_labels = []

    for pt_label in pt_ranges:
        model_path = os.path.join(base_model_path, feature_name, pt_label, f"best_model_RF_{feature_name}_{pt_label}_balanced.joblib")
        figure_dir = f"/eos/user/l/llimadas/ML_models/random_forest/shap/figures/{pt_label}"
        os.makedirs(figure_dir, exist_ok=True)

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        model = joblib.load(model_path)
        importances = model.feature_importances_

        # using shap
        test_df = pd.read_parquet(f"/eos/user/l/llimadas/nonseq_pre-processor/{medium}/{feature_name}/{pt_label}/{pt_label}_test.parquet")

        X_test = test_df.drop(columns=['Type'])
        y = test_df['Type']

        N_SAMPLES = 500
        X_sample = X_test.sample(N_SAMPLES, random_state=42) if len(X_test) > N_SAMPLES else X_test
        X_sample_renamed = X_sample.rename(columns=rename_dict)   
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        vals = shap_values.values  
        
        print(vals.ndim)  # debug
        
        if vals.ndim == 3 and vals.shape[2] == 2:
            vals = vals[:, :, 1]
        
        shap_values_fixed = shap.Explanation(
            values=vals,
            base_values=shap_values.base_values,
            data=X_sample,
            feature_names=X_sample_renamed.columns
        )
        
        # beeswarm plot
        plt.rc('font', family='serif')
        shap.plots.beeswarm(shap_values_fixed, show=False)
        plt.title(f"$p_T \in$ [{pt_label.replace('_', ', ')}] GeV", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f"shap_beeswarm_{feature_name}_{pt_label}_{medium}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(figure_dir, f"shap_beeswarm_{feature_name}_{pt_label}_{medium}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # bar plot
        plt.rc('font', family='serif')
        shap.plots.bar(shap_values_fixed, show=False)
        plt.title(f"$p_T \in$ [{pt_label.replace('_', ', ')}] GeV", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f"shap_bar_{feature_name}_{pt_label}_{medium}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(figure_dir, f"shap_bar_{feature_name}_{pt_label}_{medium}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
        
        

    
print("\nSuccessfully finished!")

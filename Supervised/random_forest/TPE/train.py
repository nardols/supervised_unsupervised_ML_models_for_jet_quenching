import os
import numpy as np
import pandas as pd
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import json
import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="R.F. training for jet classification.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Type of medium used in Jewel: default/vusp")
    parser.add_argument("--features", type=str, choices=['softdrop', 'shape', 'substructures'], required=True, help="Set of substructures")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range in the format ptmin_ptmax")
    parser.add_argument("--balance", type=str, choices=["on", "off"], default="on", help="use balanced dataset only in train if on")
    return parser.parse_args()

# Base directories
args = parse_args()
medium = args.medium
feature_set = args.features
pt_label = args.pt_label
use_balancing = args.balance == "on"
print(f"Balance flag: {args.balance} (applies only to TRAIN datasets)\n")


# Feature sets
# this code was adapted to run in parallel, but using the old structure. thus it can be easily adapted to run all configurations simultaneously

feature_sets = {
    'softdrop': ['zg', 'Rg', 'kg', 'SD_mass'],
    
    'shape': ['deltaR_TD', 'deltaR_ktD', 'SD_tau2tau1', 'SD_mz2', 'SD_ptd', 'zg_TD', 'kappa_zD', 'zg_ktD', 'SD_tau2', 'kappa_TD', 'zg', 'zg_zD'],
    
    'substructures': ['zg', 'Rg', 'kg', 'nSD', 'mass', 'mz2', 'mr', 'mr2', 'rz', 'r2z', 'ptd',
       'jetcharge03', 'jetcharge05', 'jetcharge07', 'jetcharge10', 'tau1',
       'tau2', 'tau3', 'tau4', 'tau5', 'tau2tau1', 'tau3tau2', 'kappa_TD',
       'kappa_ktD', 'kappa_zD', 'zg_TD', 'zg_ktD', 'zg_zD', 'deltaR_TD',
       'deltaR_ktD', 'deltaR_zD', 'SD_mass', 'SD_mz2', 'SD_mr', 'SD_mr2', 'SD_rz', 'SD_r2z',
       'SD_ptd', 'SD_jetcharge03', 'SD_jetcharge05', 'SD_jetcharge07',
       'SD_jetcharge10', 'SD_tau1', 'SD_tau2', 'SD_tau3', 'SD_tau4', 'SD_tau5',
       'SD_tau2tau1', 'SD_tau3tau2']
}

feature_sets = {feature_set: feature_sets[feature_set]}

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

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 600, 100),
    'max_depth': hp.choice('max_depth', [5, 10, 15, 20, 30]),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 8, 1),  
    'max_features': hp.choice('max_features', ['sqrt', 'auto']),
    'bootstrap': hp.choice('bootstrap', [True, False])
}


print('**********************************************')
print('         RANDOM FOREST TRAINING STARTED       ')
print('**********************************************\n')
print(f'Medium: [{medium}]\n')

# Main loop
for pt_label, (pt_min, pt_max) in pt_ranges.items():
    for feature_name, columns in feature_sets.items():

        print(f"Training model: {feature_name}, pT: {pt_label}")
        
        # Path to access the dataset
        base_dir = f"/sampa/llimadas/nonseq_pre-processor/{medium}/{feature_name}/{pt_label}/"
        file_path = os.path.join(base_dir, f"{'balanced' if use_balancing else ''}", f"{pt_label}_train{'_balanced' if use_balancing else ''}.parquet") # training dataset
        val_path = os.path.join(base_dir, f"{pt_label}_val.parquet")
        
        # path to save the model
        model_dir = f"/sampa/llimadas/ML_models/random_forest/models/{medium}/{feature_name}/{pt_label}" # path to save models
        os.makedirs(model_dir, exist_ok=True)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        data_train = pd.read_parquet(file_path)
        X_train = data_train.drop(columns=['Type'])
        y_train = data_train['Type']

        data_val = pd.read_parquet(val_path)
        X_val = data_val.drop(columns=['Type'])
        y_val = data_val['Type']

        def objective(params):
            params['n_estimators'] = int(params['n_estimators'])
            params['min_samples_split'] = int(params['min_samples_split'])
            params['min_samples_leaf'] = int(params['min_samples_leaf'])

            model = RandomForestClassifier(**params, random_state=SEED, n_jobs=1)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val) # using validation data

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(
            fn = objective,
            space=space,
            algo=tpe.suggest,
            max_evals = 10,
            trials = trials,
            rstate = np.random.default_rng(SEED)
        )

        best_params = {
            'n_estimators': int(best['n_estimators']),
            'max_depth': [5, 10, 15, 20, 30][best['max_depth']],
            'min_samples_split': int(best['min_samples_split']),
            'min_samples_leaf': int(best['min_samples_leaf']),
            'bootstrap': [True, False][best['bootstrap']],
            'max_features': ['sqrt', 'auto'][best['max_features']],
            'random_state': 42,
            'n_jobs': 1
        }

        # Model
        rf_model = RandomForestClassifier(**best_params)  # training with the best parameters
        rf_model.fit(X_train, y_train)

        # Save model and hyperparameters
        hyper_filename = os.path.join(model_dir, f"hyperparameters_RF_{feature_name}_{pt_label}{'_balanced' if use_balancing else ''}.txt")
        model_filename = os.path.join(model_dir, f"best_model_RF_{feature_name}_{pt_label}{'_balanced' if use_balancing else ''}.joblib")
        joblib.dump(rf_model, model_filename)

        with open(hyper_filename, 'w') as file:
            json.dump(best_params, file, indent=4)

        print(f"Model saved in: {model_filename}")

print("\nTraining finished.")



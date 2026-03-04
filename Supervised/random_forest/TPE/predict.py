# -------------------------------------------------------------------------- USAGE ----------------------------------------------------------------------------- #
#    python predict.py --mode <train/val/test> --medium <default/vusp> --features <softdrop/shape/substructures> --pt_label <ptmin_ptmax> [--inverter on|off]    #
# -------------------------------------------------------------------------- ----- ----------------------------------------------------------------------------- #


import os
import joblib
import pandas as pd
import argparse
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Prediction for jet classification using Random Forest.")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="test", help="Dataset to be predicted: train / val / test")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Target medium of the dataset used for prediction.")
    parser.add_argument("--features", type=str, choices=['softdrop', 'shape', 'substructures'], required=True, help="Feature set.")
    parser.add_argument("--pt_label", type=str, required=True, help="pT range in the format ptmin_ptmax.")
    parser.add_argument("--inverter", type=str, choices=['on', 'off'], default='off', help="If 'on', use the model trained on the opposite medium.")
    parser.add_argument("--balance", type=str, choices=["on", "off"], default="on", help="use model trained on balanced dataset only in train if on")
    return parser.parse_args()

def invert_medium(medium: str) -> str:
    return "vusp" if medium == "default" else "default"

args = parse_args()
mode = args.mode
data_medium = args.medium              # medium of the DATA used for prediction
feature_set = args.features
pt_label = args.pt_label
inverter_tag = args.inverter
use_balancing = args.balance == "on"
print(f"Balance flag: {args.balance} (applies only to TRAIN datasets)\n")

# defining training medium (where the model was trained)
train_medium = invert_medium(data_medium) if inverter_tag == 'on' else data_medium

print('**********************************************')
print('   RANDOM FOREST PREDICTION CALCULATION       ')
print('**********************************************\n')
print(f"Dataset: [{mode.upper()}] | Data medium: [{data_medium}] | Model trained on: [{train_medium}] | Inverter: [{inverter_tag}]\n")

scenario = "inverted" if inverter_tag == "on" else "standard"

model_path = f"/sampa/llimadas/ML_models/random_forest/models/{train_medium}/{feature_set}/{pt_label}/best_model_RF_{feature_set}_{pt_label}{'_balanced' if use_balancing else ''}.joblib"

base_dir = f"/sampa/llimadas/nonseq_pre-processor/{data_medium}/{feature_set}/{pt_label}/"
if use_balancing and mode == "train":
    data_path = os.path.join(base_dir, "balanced", f"{pt_label}_{mode}_balanced.parquet")
else:
    data_path = os.path.join(base_dir, f"{pt_label}_{mode}.parquet")

# outputs dir
output_dir = f"/sampa/llimadas/ML_models/random_forest/predictions/{scenario}/{feature_set}/{pt_label}"
os.makedirs(output_dir, exist_ok=True)
output_filename = f"{pt_label}_{mode}_predictions_{feature_set}_train-{train_medium}_data-{data_medium}{'_balanced' if use_balancing else ''}.parquet"
output_path = os.path.join(output_dir, output_filename)


if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit()
if not os.path.exists(data_path):
    print(f"Data not found: {data_path}")
    exit()


model = joblib.load(model_path)
data = pd.read_parquet(data_path)
X = data.drop(columns=["Type"])
y_true = data["Type"]


y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# Save results
result_df = X.copy()
result_df["True"] = y_true
result_df["Pred"] = y_pred
result_df["Prob"] = y_proba
result_df.to_parquet(output_path, index=False)
print(f"Results saved to: {output_path}")


print(classification_report(y_true, y_pred, digits=3))

print("\nPREDICTIONS COMPLETED.")

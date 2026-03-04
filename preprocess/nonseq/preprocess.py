### USAGE: python preprocess.py --mode <train/val/test> --type <features/PCA> --medium <default/vusp> [--smote on/off] [--scaler on/off]

import os
import argparse
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser(description="data preprocessing for jet classification.")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], required=True, help="mode: train / val / test")
    parser.add_argument("--type", type=str, choices=["features", "PCA"], required=True, help="preprocessing type: features/pca")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="type of medium used in jewel: default/vusp")
    parser.add_argument("--balance", type=str, choices=["on", "off"], default="on", help="balance dataset only in train if on")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], default="off", help="apply StandardScaler if on")
    return parser.parse_args()

def main():
    print("****************************************************\n************RUNNING PRE-PROCESSOR CODE ************\n***************************************************\n")
    
    args = parse_args()
    mode = args.mode
    inclusive_max = "on"

    # flags
    use_balancing = args.balance == "on"
    use_scaler = args.scaler == "on"

    INPUT_DIR = "/sampa/llimadas/data/"

    # separate features
    if args.type == 'PCA':
        set = {'PCA': [5, 10, 15]}
    elif args.type == 'features':
        set = {
            'softdrop': ['zg', 'Rg', 'kg', 'SD_mass'],
            'shape': ['deltaR_TD', 'deltaR_ktD', 'SD_tau2tau1', 'SD_mz2', 'SD_ptd', 'zg_TD', 'kappa_zD', 'zg_ktD', 'SD_tau2', 'kappa_TD', 'zg', 'zg_zD'],
            'substructures': ['zg','Rg', 'kg', 'nSD', 'mass', 'mz2', 'mr', 'mr2', 'rz', 'r2z', 'ptd','jetcharge03', 'jetcharge05', 'jetcharge07', 'jetcharge10',
                              'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau2tau1', 'tau3tau2', 'kappa_TD', 'kappa_ktD', 'kappa_zD', 'zg_TD', 'zg_ktD', 'zg_zD',
                              'deltaR_TD', 'deltaR_ktD', 'deltaR_zD', 'SD_mass', 'SD_mz2', 'SD_mr', 'SD_mr2', 'SD_rz', 'SD_r2z', 'SD_ptd', 'SD_jetcharge03',
                              'SD_jetcharge05', 'SD_jetcharge07', 'SD_jetcharge10', 'SD_tau1', 'SD_tau2', 'SD_tau3', 'SD_tau4', 'SD_tau5',
                              'SD_tau2tau1', 'SD_tau3tau2']
        }

    pt_ranges = {
        '40_60': (40, 60),
        '60_80': (60, 80),
        '80_120': (80, 120),
        '120_200': (120, 200),
        '200_400': (200, 400),
        '80_250': (80, 250),
    }

    medium = args.medium

    #----------------#
    # main loop #
    #----------------#

    for pt_label, (pt_min, pt_max) in pt_ranges.items():
        for feat_name, columns in set.items():

            OUTPUT_DIR = f"/sampa/llimadas/nonseq_pre-processor/{medium}/{feat_name}/{pt_label}/"
            SCALER_DIR = f"/sampa/llimadas/nonseq_pre-processor/scalars/{medium}/{feat_name}/{pt_label}/"
        
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(SCALER_DIR, exist_ok=True)
            
            print(f"#--------------------------------------------------------------#\n[{mode.upper()}] | pre-processing {feat_name} | pT: {pt_label}\n#--------------------------------------------------------------#")

            file_pp = os.path.join(INPUT_DIR, f"output_pp_full_nobkgsub_article-marco_{mode}.parquet")
            file_pbpb = os.path.join(INPUT_DIR, f"output_PbPb_{medium}_recoils-on_full_bkgsub_article-marco_{mode}.parquet")

            df_pp = pd.read_parquet(file_pp, engine="pyarrow")
            df_pbpb = pd.read_parquet(file_pbpb, engine="pyarrow")
            
            for var in ['Rg', 'zg', 'SD_mass', 'kg']:
                if var in df_pp.columns:
                    before = len(df_pp)
                    df_pp = df_pp[df_pp[var] >= 0]
                    print(f"pp removed {before - len(df_pp)} rows by {var}>=0")
                if var in df_pbpb.columns:
                    before = len(df_pbpb)
                    df_pbpb = df_pbpb[df_pbpb[var] >= 0]
                    print(f"pbpb removed {before - len(df_pbpb)} rows by {var}>=0")

            # pt filter 
            if inclusive_max:
                pp_mask = (df_pp['pt'] >= pt_min) & (df_pp['pt'] <= pt_max)
                pb_mask = (df_pbpb['pt'] >= pt_min) & (df_pbpb['pt'] <= pt_max)
            else:
                pp_mask = (df_pp['pt'] >= pt_min) & (df_pp['pt'] < pt_max)
                pb_mask = (df_pbpb['pt'] >= pt_min) & (df_pbpb['pt'] < pt_max)

            df_pp = df_pp[pp_mask].copy()
            df_pbpb = df_pbpb[pb_mask].copy()

            # feature selection vs pca inputs
            if args.type != 'PCA':
                df_pp = df_pp[columns].copy()
                df_pbpb = df_pbpb[columns].copy()
            else:
                cols_to_drop = ['ievt', 'ijet', 'evwt', 'pt', 'eta', 'rapidity', 'phi', 'nconst',
                                'SD_pt', 'SD_eta', 'SD_rapidity', 'SD_phi', 'SD_nconst',
                                'depth', 'z', 'delta', 'kperp', 'minv']
                df_pp.drop(columns=[c for c in cols_to_drop if c in df_pp.columns], inplace=True)
                df_pbpb.drop(columns=[c for c in cols_to_drop if c in df_pbpb.columns], inplace=True)
            
            # drop duplicates 
            df_pp = df_pp.drop_duplicates()
            df_pbpb = df_pbpb.drop_duplicates()

            # imputation
            imputer_path = os.path.join(SCALER_DIR, f"{pt_label}_imputer.save")
            if mode == "train":
                imputer = SimpleImputer(strategy="median")
                imputer.fit(pd.concat([df_pp, df_pbpb], ignore_index=True))
                joblib.dump(imputer, imputer_path)
                print(f"imputer saved: {imputer_path}")
            else:
                if not os.path.exists(imputer_path):
                    print(f"imputer not found: {imputer_path}")
                    continue
                imputer = joblib.load(imputer_path)
                print(f"imputer loaded: {imputer_path}")

            df_pp = pd.DataFrame(imputer.transform(df_pp), columns=df_pp.columns, index=df_pp.index)
            df_pbpb = pd.DataFrame(imputer.transform(df_pbpb), columns=df_pbpb.columns, index=df_pbpb.index)

            # class labels
            df_pp["Type"] = 0
            df_pbpb["Type"] = 1
            
            # combine
            df_combined = pd.concat([df_pp, df_pbpb], ignore_index=True)
            
            # split features/labels
            X = df_combined.drop(columns=["Type"])
            y = df_combined["Type"]
            
            # scaler
            if use_scaler:
                scaler_path = os.path.join(SCALER_DIR, f"{pt_label}_scaler.save")
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

                if mode == "train":
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    joblib.dump(scaler, scaler_path)
                    print(f"scaler saved: {scaler_path}")
                else:
                    if not os.path.exists(scaler_path):
                        print(f"scaler not found: {scaler_path}")
                        continue
                    scaler = joblib.load(scaler_path)
                    X_scaled = scaler.transform(X)
                    print(f"scaler loaded: {scaler_path}")
            else:
                print("Scaler disabled, using raw features.")
                X_scaled = X.values

            if args.type == "PCA":
                for n_components in set[feat_name]:
                    pca_path = os.path.join(SCALER_DIR, f"{pt_label}_pca_{n_components}.save")

                    if mode == "train":
                        pca = PCA(n_components=n_components, random_state=42)
                        X_reduced = pca.fit_transform(X_scaled)
                        joblib.dump(pca, pca_path)
                        print(f"pca ({n_components} comps) saved: {pca_path}")
                    else:
                        if not os.path.exists(pca_path):
                            print(f"pca not found: {pca_path}")
                            continue
                        pca = joblib.load(pca_path)
                        X_reduced = pca.transform(X_scaled)
                        print(f"pca ({n_components} comps) loaded: {pca_path}")

                    df_out = pd.DataFrame(X_reduced, columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])])
                    df_out["Type"] = y.reset_index(drop=True)

                    output_path = os.path.join(OUTPUT_DIR, f"{pt_label}_{mode}_pca{n_components}.parquet")
                    df_out.to_parquet(output_path, engine="pyarrow", index=False)
                    print(f"file saved: {output_path}")
            else:
                df_combined = pd.DataFrame(X_scaled, columns=df_combined.drop(columns=["Type"]).columns)
                df_combined["Type"] = y

                if mode == "train" and use_balancing:
                    print("applying undersampling (train only)")
                    rus = RandomUnderSampler(random_state=42)
                    X_ = df_combined.drop(columns=["Type"])
                    y_ = df_combined["Type"]
                    X_bal, y_bal = rus.fit_resample(X_, y_)
                    df_combined = pd.DataFrame(X_bal, columns=X_.columns)
                    df_combined["Type"] = y_bal
                    print("\nUndersampling applied.")

                    save_dir = OUTPUT_DIR if not use_balancing else os.path.join(OUTPUT_DIR, "balanced")
                    os.makedirs(save_dir, exist_ok=True)
                    output_path = os.path.join(save_dir, f"{pt_label}_{mode}{'_balanced' if use_balancing else ''}.parquet")
                    df_combined.to_parquet(output_path, engine="pyarrow", index=False)
                    print(f"file saved: {output_path}")

                else:
                    output_path = os.path.join(OUTPUT_DIR, f"{pt_label}_{mode}.parquet")
                    df_combined.to_parquet(output_path, engine="pyarrow", index=False)
                    print(f"file saved: {output_path}")

    print(f"\nend of preprocessing in mode [{mode.upper()}].")

if __name__ == "__main__":
    main()

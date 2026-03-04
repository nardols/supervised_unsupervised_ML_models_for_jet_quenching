import os
import argparse
import pandas as pd
import uproot
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import awkward as ak


## FUNCTIONS ##

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing of sequences for jet classification (LSTM).")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], required=True, help="Mode: train / val / test")
    parser.add_argument("--scaler", type=str, choices=["on", "off"], required=True, help="Apply scaler to the sequences: on / off")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Type of medium used in Jewel: default/vusp")
    return parser.parse_args()

def load_root_sequences(file_path):
    with uproot.open(file_path) as f:
        tree = f["jetprops"]
        branches = ['z', 'delta', 'kperp', 'minv', 'pt']
        arrays = tree.arrays(branches, library="ak")
        x_t_list = [
            [[z, d, k, m] for z, d, k, m in zip(zs, ds, ks, ms)]
            for zs, ds, ks, ms in zip(arrays["z"], arrays["delta"], arrays["kperp"], arrays["minv"])
        ]
        pt = ak.to_numpy(arrays["pt"])
        df = pd.DataFrame({"pt": pt, "x_t": x_t_list})
    return df

def is_valid_sequence(seq):
    return all(
        (step[0] >= 0 and step[1] >= 0 and step[2] >= 0 and step[3] >= 0)
        for step in seq if step is not None
    )

def drop_duplicate_sequences(df):
    df['x_t_aux'] = df['x_t'].apply(lambda seq: tuple(tuple(step) for step in seq))
    df = df.drop_duplicates(subset=['x_t_aux'])
    return df.drop(columns=['x_t_aux'])


##########


def main():
    print("****************************************************")
    print("************ RUNNING SEQUENCE PREPROCESSOR *********")
    print("****************************************************\n")

    args = parse_args()
    mode = args.mode
    medium = args.medium
    scaler_tag = args.scaler
    inclusive_max = "on"

    INPUT_DIR = f"/sampa/llimadas/data"

    pt_ranges = {
        '40_60': (40, 60),
        '60_80': (60, 80),
        '80_120': (80, 120),
        '120_200': (120, 200),
        '200_400': (200, 400),
        '80_250': (80, 250),
    }

    for pt_label, (pt_min, pt_max) in pt_ranges.items():
        print(f"#--------------------------------------------------------------#")
        print(f"[{mode.upper()}] | Pre-processing LSTM sequences | pT: {pt_label}")
        print(f"#--------------------------------------------------------------#")

        file_pp = os.path.join(INPUT_DIR, f"output_pp_full_nobkgsub_article-marco_{mode}.root")
        file_pbpb = os.path.join(INPUT_DIR, f"output_PbPb_{medium}_recoils-on_full_bkgsub_article-marco_{mode}.root")

        df_pp = load_root_sequences(file_pp)
        df_pbpb = load_root_sequences(file_pbpb)

        n_pp_0 = len(df_pp)
        n_pb_0 = len(df_pbpb)
        df_pp = drop_duplicate_sequences(df_pp)
        df_pbpb = drop_duplicate_sequences(df_pbpb)

        if inclusive_max:
            df_pp = df_pp[(df_pp['pt'] >= pt_min) & (df_pp['pt'] <= pt_max)].copy()
            df_pbpb = df_pbpb[(df_pbpb['pt'] >= pt_min) & (df_pbpb['pt'] <= pt_max)].copy()
        else:
            df_pp = df_pp[(df_pp['pt'] >= pt_min) & (df_pp['pt'] < pt_max)].copy()
            df_pbpb = df_pbpb[(df_pbpb['pt'] >= pt_min) & (df_pbpb['pt'] < pt_max)].copy()

        n_pp_before = len(df_pp)
        n_pb_before = len(df_pbpb)
        df_pp = df_pp[df_pp['x_t'].apply(is_valid_sequence)].copy()
        df_pbpb = df_pbpb[df_pbpb['x_t'].apply(is_valid_sequence)].copy()

        df_pp.drop(columns=['pt'], inplace=True)
        df_pbpb.drop(columns=['pt'], inplace=True)

        df_pp['Type'] = 0
        df_pbpb['Type'] = 1

        df_combined = pd.concat([df_pp, df_pbpb], ignore_index=True)

        OUTPUT_DIR = f"/sampa/llimadas/seq_pre-processor/processed_data/results/{medium}/{mode}/{pt_label}/"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if scaler_tag == "on":
            SCALER_DIR = f"/sampa/llimadas/seq_pre-processor/processed_data/scalers/{medium}/train/{pt_label}/"
            os.makedirs(SCALER_DIR, exist_ok=True)

            z_vals     = [ [step[0] for step in seq] for seq in df_combined['x_t'] ]
            delta_vals = [ [step[1] for step in seq] for seq in df_combined['x_t'] ]
            kperp_vals = [ [step[2] for step in seq] for seq in df_combined['x_t'] ]
            m_vals     = [ [step[3] for step in seq] for seq in df_combined['x_t'] ]

            def scale_and_replace(values, name):
                scaler_path = os.path.join(SCALER_DIR, f"{pt_label}_{name}_scaler.save")
                if mode == "train":
                    scaler = StandardScaler()
                    flat_data = np.concatenate([v for v in values if len(v) > 0]).reshape(-1, 1)
                    scaler.fit(flat_data)
                    joblib.dump(scaler, scaler_path)
                    print(f"Scaler saved: {scaler_path}")
                else:
                    if not os.path.exists(scaler_path):
                        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
                    scaler = joblib.load(scaler_path)
                    print(f"Scaler loaded: {scaler_path}")
                scaled = [
                    scaler.transform(np.array(v).reshape(-1, 1)).flatten().tolist()
                    if len(v) > 0 else []
                    for v in values
                ]
                return scaled

            z_scaled     = scale_and_replace(z_vals, "z")
            delta_scaled = scale_and_replace(delta_vals, "delta")
            kperp_scaled = scale_and_replace(kperp_vals, "kperp")
            m_scaled     = scale_and_replace(m_vals, "minv")

            df_combined['x_t'] = [
                [[z, d, k, m] for z, d, k, m in zip(zseq, dseq, kseq, mseq)]
                for zseq, dseq, kseq, mseq in zip(z_scaled, delta_scaled, kperp_scaled, m_scaled)
            ]

        before_empty = len(df_combined)
        df_combined = df_combined[df_combined['x_t'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        output_path = os.path.join(OUTPUT_DIR, f"{pt_label}_{mode}_scaler-{scaler_tag}.parquet")
        df_combined.to_parquet(output_path, engine="pyarrow", index=False)
        print(f"File saved: {output_path}\n")

    print(f"\nEnd of sequence preprocessing in mode [{mode.upper()}].")

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import argparse

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score, silhouette_score,
    davies_bouldin_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append("/eos/user/l/llimadas/unsupervised/LSTM-AutoEncoder/TPE/model")
from kmeans import KMeans_SEMI


# ======================================
# ARGUMENT PARSER
# ======================================
def parse_args():
    parser = argparse.ArgumentParser(description="AUTOENCODER FOR UNSUPERVISED CLUSTERING")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True,
                        help="Medium type used in JEWEL.")
    return parser.parse_args()


# ======================================
# GPU CONFIG
# ======================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU successfully configured.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")
else:
    print("No GPU detected. Running on CPU.")


# PLOT STYLE
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



# MAIN CONFIG
args = parse_args()
medium = args.medium

dataset_dir = f"/eos/user/l/llimadas/nonseq_pre-processor/{medium}/"

os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

feature_sets = {
    'softdrop': ['zg', 'Rg', 'kg', 'SD_mass'],
}

pt_intervals = {
    '40_60': (40, 60),
    '200_400': (200, 400),
    '80_250': (80, 250)
}

latent_dims = [15]
sample_size = 2500


# ======================================
# MAIN LOOP
# ======================================
for pt_label, (pt_min, pt_max) in pt_intervals.items():
    for feature_name, columns in feature_sets.items():

        file_path = os.path.join(
            dataset_dir, feature_name, pt_label, f"{pt_label}_train_balanced.parquet"
        )

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"\n===== pT Range: {pt_label} | Feature Set: {feature_name} =====")

        data = pd.read_parquet(file_path)
        X = data[columns].values
        y = data['Type'].values  # 0 unquenched, 1 quenched

        X = tf.convert_to_tensor(X, dtype=tf.float32)

        for latent_dim in latent_dims:
            print(f"\n--- Latent Dim: {latent_dim} ---")

            input_dim = X.shape[1]
            input_layer = Input(shape=(input_dim,))

            # ++++++++++++++++++++++++++
            # AUTOENCODER ARCHITECTURE
            # +++++++++++++++++++++++++
            
            encoded = Dense(256, activation='relu')(input_layer)
            encoded = Dropout(0.2)(encoded)
            encoded = Dense(128, activation='relu')(encoded)
            encoded = Dropout(0.2)(encoded)
            encoded = Dense(64, activation='relu')(encoded)
            encoded = Dense(32, activation='relu')(encoded)
            encoded = Dense(latent_dim, activation='relu')(encoded)

            decoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(64, activation='relu')(decoded)
            decoded = Dense(128, activation='relu')(decoded)
            decoded = Dense(256, activation='relu')(decoded)
            decoded = Dense(input_dim, activation='linear')(decoded)

            autoencoder = Model(inputs=input_layer, outputs=decoded)
            encoder = Model(inputs=input_layer, outputs=encoded)
            autoencoder.compile(optimizer='adam', loss='mse')

            print("Training autoencoder...")
            autoencoder.fit(
                X, X,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_split=0.2,
                verbose=1,
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
            )

            # SAVING MODEL
            model_name = f"models/autoencoder_{feature_name}_{pt_label}_{latent_dim}dim.joblib"
            joblib.dump(autoencoder, model_name)
            print(f"Model saved: {model_name}")

            # ENCODE DATA
            X_encoded = encoder.predict(X)

            
            # ----------- PCA 2D -----------
            pca2 = PCA(n_components=2)
            X_pca2 = pca2.fit_transform(X_encoded)
            
            fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=300)
            
            colors_latent = np.where(y == 0, "#1f77b4", "#d62728")
            
            ax2.scatter(
                X_pca2[:, 0], X_pca2[:, 1],
                c=colors_latent,
                s=14, edgecolor='k', linewidth=0.2, alpha=0.8
            )
            
            #ax2.set_title(
            #    rf"PCA-2D latent space ({pt_label} GeV) — {latent_dim}D → 2D "
            #    rf"\nExplained variance: {pca2.explained_variance_ratio_[0]:.2f}, "
            #    rf"{pca2.explained_variance_ratio_[1]:.2f}",
            #    fontsize=14
            #)
            
            ax2.set_xlabel("PCA 1", fontsize=22)
            ax2.set_ylabel("PCA 2", fontsize=22)
            #ax2.grid(alpha=0.25)
            
            ax2.legend(
                handles=[
                    plt.Line2D([], [], marker='o', color="#1f77b4", linestyle='',
                               label="Unquenched", markersize=6),
                    plt.Line2D([], [], marker='o', color="#d62728", linestyle='',
                               label="Quenched", markersize=6),
                ],
                frameon=True, framealpha=0.9, facecolor='white', fontsize=14
            )
            
            fig2.tight_layout()
            fig2_name = f"figures/latent_space_2D_{feature_name}_{medium}_{pt_label}_{latent_dim}dim.pdf"
            plt.savefig(fig2_name, bbox_inches='tight')
            plt.close()
            print(f"Saved latent PCA-2D: {fig2_name}")
            
            
            # ----------- PCA 3D -----------
            
            pca3 = PCA(n_components=3)
            X_pca3 = pca3.fit_transform(X_encoded)
            
            fig3 = plt.figure(figsize=(8, 7), dpi=300)
            ax3 = fig3.add_subplot(111, projection='3d')
            
            ax3.scatter(
                X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2],
                c=colors_latent,
                s=18, edgecolor='k', linewidth=0.2, alpha=0.8
            )
            
            #ax3.set_title(
            #    rf"PCA-3D latent space ({pt_label} GeV) — {latent_dim}D → 3D "
            #    rf"\nExplained variance: {pca3.explained_variance_ratio_[0]:.2f}, "
            #    rf"{pca3.explained_variance_ratio_[1]:.2f}, "
            #    rf"{pca3.explained_variance_ratio_[2]:.2f}",
            #    fontsize=14
            #)
            
            ax3.set_xlabel("PCA 1", fontsize=22)
            ax3.set_ylabel("PCA 2", fontsize=22)
            ax3.set_zlabel("PCA 3", fontsize=22)
            
            # -----------lLegend for Quenched / Unquenched -----------
            
            legend_handles = [
                plt.Line2D([], [], marker='o', color="#1f77b4", linestyle='',
                           label="Unquenched", markersize=8),
                plt.Line2D([], [], marker='o', color="#d62728", linestyle='',
                           label="Quenched", markersize=8),
            ]
            
            ax3.legend(
                handles=legend_handles,
                frameon=True,
                framealpha=0.9,
                facecolor='white',
                fontsize=14,
                loc='upper right'
            )
            
            fig3.tight_layout()
            fig3_name = f"figures/latent_space_3D_{feature_name}_{medium}_{pt_label}_{latent_dim}dim.pdf"
            plt.savefig(fig3_name, bbox_inches='tight')
            plt.close()
            
            print(f"Saved latent PCA-3D: {fig3_name}")


            # CLUSTERING (KMEANS ONLY)
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=50)
            kmeans_labels = kmeans.fit_predict(X_encoded)

            # METRICS
            metrics_text = (
                f"Feature set: {feature_name} | pT: {pt_label} | Latent dim: {latent_dim}\n\n"
                f"KMeans Metrics:\n"
            )

            ari = adjusted_rand_score(y, kmeans_labels)
            nmi = normalized_mutual_info_score(y, kmeans_labels)
            h = homogeneity_score(y, kmeans_labels)
            c = completeness_score(y, kmeans_labels)
            v = v_measure_score(y, kmeans_labels)
            sil = silhouette_score(X_encoded, kmeans_labels)
            db = davies_bouldin_score(X_encoded, kmeans_labels)

            metrics_text += (
                f"  ARI: {ari:.4f}\n"
                f"  NMI: {nmi:.4f}\n"
                f"  Homogeneity: {h:.4f}\n"
                f"  Completeness: {c:.4f}\n"
                f"  V-Measure: {v:.4f}\n"
                f"  Silhouette: {sil:.4f}\n"
                f"  Davies-Bouldin: {db:.4f}\n"
            )

            print(metrics_text)

            # SAVE METRICS
            metrics_path = f"metrics/metrics_{medium}_{feature_name}_{pt_label}_{latent_dim}.txt"
            with open(metrics_path, "w") as f:
                f.write(metrics_text)

            # =====================================
            # PCA VISUALIZATION
            # =====================================
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X_encoded)

            num_samples = min(sample_size, X_vis.shape[0])
            indices = np.random.choice(X_vis.shape[0], num_samples, replace=False)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

            # Colors
            colors_true = np.where(y == 0, "#1f77b4", "#d62728")
            colors_km = np.where(kmeans_labels == 0, "#1f77b4", "#d62728")

            # ---------- KMEANS CLUSTERS ----------
            axes[0].scatter(
                X_vis[indices, 0], X_vis[indices, 1],
                c=colors_km[indices],
                s=18, edgecolor='k', linewidth=0.2, alpha=0.8
            )
            axes[0].set_title("KMeans Clusters", fontsize=24)
            axes[0].set_xlabel("PCA 1", fontsize=22)
            axes[0].set_ylabel("PCA 2", fontsize=22)
            #axes[0].grid(alpha=0.2)

            axes[0].legend(
                handles=[
                    plt.Line2D([], [], marker='o', color="#1f77b4", linestyle='',
                               label="Cluster 0", markersize=8),
                    plt.Line2D([], [], marker='o', color="#d62728", linestyle='',
                               label="Cluster 1", markersize=8),
                ],
                frameon=True, framealpha=0.9, facecolor='white',
                fontsize=18
            )

            # ---------- TRUE LABELS PLOT ----------
            axes[1].scatter(
                X_vis[indices, 0], X_vis[indices, 1],
                c=colors_true[indices],
                s=18, edgecolor='k', linewidth=0.2, alpha=0.8
            )
            axes[1].set_title("True Labels", fontsize=24)
            axes[1].set_xlabel("PCA 1", fontsize=18)
            axes[1].set_ylabel("PCA 2", fontsize=18)
            #axes[1].grid(alpha=0.2)

            axes[1].legend(
                handles=[
                    plt.Line2D([], [], marker='o', color="#1f77b4", linestyle='',
                               label="Unquenched", markersize=8),
                    plt.Line2D([], [], marker='o', color="#d62728", linestyle='',
                               label="Quenched", markersize=8),
                ],
                frameon=True, framealpha=0.9, facecolor='white',
                fontsize=18
            )


            #plt.suptitle(
            #    rf"$p_{{T}}$ range: {pt_label.split('_')[0]} - {pt_label.split('_')[-1]}  GeV — {latent_dim} latent dimensions",
            #    fontsize=20
            #)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            fig_name = f"figures/clusters_{feature_name}_{medium}_{pt_label}_{latent_dim}dim.pdf"
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close()

            print(f"Figure saved: {fig_name}\n")

            del X_encoded
            K.clear_session()
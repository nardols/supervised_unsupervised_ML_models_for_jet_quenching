import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib as mpl
import argparse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="OPTIMIZED K-MEANS EXECUTION.")
    parser.add_argument("--medium", type=str, choices=["default", "vusp"], required=True, help="Medium type used in JEWEL: default/vusp")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 to use all cores)")
    parser.add_argument("--sampling_size", type=int, default=2500, help="Sampling size for visualization (reduces plot time)")
    parser.add_argument("--skip_plots", action="store_true", default=False, help="Skip plot generation to speed up analysis")
    return parser.parse_args()


args = parse_args()

# Optimized settings
base_dir = "/eos/user/l/llimadas/nonseq_pre-processor"
medium = args.medium
scaler_tag = "scaler-on"
mode = "train"
n_jobs = args.n_jobs if args.n_jobs != -1 else cpu_count()
sampling_size = args.sampling_size
skip_plots = args.skip_plots

print(f"Using {n_jobs} cores for parallelization")

pt_ranges = {
    '40_60': (40, 60),
    '200_400': (200, 400),
    '80_250': (80, 250)
}

feature_sets = {
    'softdrop': ['zg', 'Rg', 'kg', 'SD_mass'],
    'shape': ['deltaR_TD', 'deltaR_ktD', 'SD_tau2tau1', 'SD_mz2', 'SD_ptd', 'zg_TD', 'kappa_zD', 'zg_ktD', 'SD_tau2', 'kappa_TD', 'zg', 'zg_zD']
}

if not skip_plots:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams.update({
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'figure.titlesize': 26,
        'figure.dpi': 300
    })


def calculate_comprehensive_metrics(true_labels, predicted_labels, feature_data, kmeans_model):

    metrics = {}

    supervised_metrics = {
        'ARI': adjusted_rand_score,
        'NMI': normalized_mutual_info_score,
        'FMI': fowlkes_mallows_score,
        'Homogeneity': homogeneity_score,
        'Completeness': completeness_score,
        'V-measure': v_measure_score
    }

    for name, func in supervised_metrics.items():
        metrics[name] = func(true_labels, predicted_labels)

    metrics['Silhouette'] = silhouette_score(feature_data, predicted_labels)
    metrics['Calinski_Harabasz'] = calinski_harabasz_score(feature_data, predicted_labels)
    metrics['Davies_Bouldin'] = davies_bouldin_score(feature_data, predicted_labels)
    metrics['Inertia'] = kmeans_model.inertia_

    cm = confusion_matrix(true_labels, predicted_labels)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        metrics.update({
            'Precision': precision,
            'Recall': recall,
            'F1_Score': 2 * precision * recall / (precision + recall + eps),
            'Accuracy': (tp + tn) / cm.sum(),
            'Specificity': tn / (tn + fp + eps)
        })

    return metrics


def plot_comprehensive_analysis_optimized(data, clusters, true_labels, kmeans, feature_name, columns, pt_min, pt_max, metrics, outdir, pt_label):
    os.makedirs(outdir, exist_ok=True)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams.update({
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'figure.titlesize': 26,
        'figure.dpi': 300
    })

    # balanced sampling
    n_samples = min(sampling_size, data.shape[0])
    if n_samples < data.shape[0]:
        idx0 = np.where(clusters == 0)[0]
        idx1 = np.where(clusters == 1)[0]
        n0 = min(n_samples // 2, len(idx0))
        n1 = min(n_samples // 2, len(idx1))
        s0 = np.random.choice(idx0, n0, replace=False) if n0 > 0 else np.array([], dtype=int)
        s1 = np.random.choice(idx1, n1, replace=False) if n1 > 0 else np.array([], dtype=int)
        indices = np.concatenate([s0, s1])
    else:
        indices = np.arange(data.shape[0])

    feature_data = data[columns].values

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(feature_data)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    fig_scatter, axs_scatter = plt.subplots(1, 2, figsize=(18, 7))
    label_colors = {0: "blue", 1: "red"}

    axs_scatter[0].scatter(data_pca[indices, 0], data_pca[indices, 1], c=[label_colors[t] for t in clusters[indices]], s=20, alpha=0.7, rasterized=True)
    axs_scatter[0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='x', s=150, label='Cluster Centers')
    axs_scatter[0].set_title(f"K-means PCA")
    axs_scatter[0].set_xlabel("PCA 1")
    axs_scatter[0].set_ylabel("PCA 2")
    handles = [
        plt.Line2D([], [], marker='o', color='w', label='Class 0', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Class 1', markerfacecolor='red', markersize=10),
        plt.Line2D([], [], marker='x', color='black', label='Cluster Centers', markersize=10)
    ]
    axs_scatter[0].legend(handles=handles)

    axs_scatter[1].scatter(data_pca[indices, 0], data_pca[indices, 1], c=[label_colors[t] for t in true_labels.iloc[indices]], s=20, alpha=0.7, rasterized=True)
    axs_scatter[1].set_title(f"Ground Truth PCA")
    axs_scatter[1].set_xlabel("PCA 1")
    axs_scatter[1].set_ylabel("PCA 2")
    handles_gt = [
        plt.Line2D([], [], marker='o', color='w', label='Unquenched', markerfacecolor='blue', markersize=10),
        plt.Line2D([], [], marker='o', color='w', label='Quenched', markerfacecolor='red', markersize=10)
    ]
    axs_scatter[1].legend(handles=handles_gt)

    fig_scatter.tight_layout()
    scatter_path_png = os.path.join(outdir, f"kmeans_scatter_{feature_name}_{pt_label}_{medium}.png")
    scatter_path_pdf = os.path.join(outdir, f"kmeans_scatter_{feature_name}_{pt_label}_{medium}.pdf")
    fig_scatter.savefig(scatter_path_pdf)
    fig_scatter.savefig(scatter_path_png, bbox_inches='tight', dpi=150)
    plt.close(fig_scatter)

    fig_cm, ax_cm = plt.subplots(1, 1, figsize=(7, 6))
    cm = confusion_matrix(true_labels, clusters)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_xlabel('Predicted Cluster')
    ax_cm.set_ylabel('True Class')
    cm_path = os.path.join(outdir, f"kmeans_confusion_matrix_{feature_name}_{pt_label}.png")
    fig_cm.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close(fig_cm)

    def fmt_opt(v, fmt=".3f"):
        try: return format(v, fmt)
        except: return str(v)

    txt = f"""Performance Metrics
======================
pT range: {pt_min}-{pt_max} GeV
Feature set: {feature_name}
Total samples: {len(data):,}

Supervised:
- ARI: {fmt_opt(metrics.get('ARI'))}
- NMI: {fmt_opt(metrics.get('NMI'))}
- F1: {fmt_opt(metrics.get('F1_Score'))}
- Precision: {fmt_opt(metrics.get('Precision'))}

Unsupervised:
- Silhouette: {fmt_opt(metrics.get('Silhouette'))}
- Calinski-Harabasz: {fmt_opt(metrics.get('Calinski_Harabasz'), '.1f')}
- Davies-Bouldin: {fmt_opt(metrics.get('Davies_Bouldin'))}
"""

    txt_path = os.path.join(outdir, f"kmeans_metrics_{feature_name}_{pt_label}_{medium}.txt")
    with open(txt_path, "w") as f:
        f.write(txt)

    return scatter_path_pdf


def run_single_experiment(pt_label, pt_min, pt_max, feature_name, columns):
    try:
        outdir = f"/eos/user/l/llimadas/unsupervised/kmeans_results/{medium}/{pt_label}/{feature_name}"
        os.makedirs(outdir, exist_ok=True)

        file_path = os.path.join(base_dir, medium, feature_name, pt_label,
                                 f"{pt_label}_{mode}_balanced.parquet")

        if not os.path.exists(file_path):
            return None

        data = pd.read_parquet(file_path, columns=columns + ['Type'])

        if not all(col in data.columns for col in columns + ['Type']):
            return None

        feature_data = data[columns].values

        kmeans = KMeans(
            n_clusters=2, random_state=42, n_init=20,
            max_iter=100, algorithm='lloyd'
        )
        clusters = kmeans.fit_predict(feature_data)

        cm_initial = confusion_matrix(data['Type'], clusters)
        if cm_initial.trace() < (cm_initial.sum() - cm_initial.trace()):
            clusters = 1 - clusters

        metrics = calculate_comprehensive_metrics(data['Type'], clusters, feature_data, kmeans)

        experiment_info = {
            'pt_range': pt_label,
            'pt_min': pt_min,
            'pt_max': pt_max,
            'feature_set': feature_name,
            'n_samples': len(data)
        }

        result = {**experiment_info, **metrics}

        if not skip_plots:
            plot_comprehensive_analysis_optimized(
                data, clusters, data['Type'], kmeans,
                feature_name, columns, pt_min, pt_max, metrics, outdir, pt_label
            )

        return result

    except Exception as e:
        print(f"Error in {pt_label}-{feature_name}: {e}")
        return None


def main():
    print("Starting K-means analysis...")

    experiments = []
    for pt_label, (pt_min, pt_max) in pt_ranges.items():
        for feature_name, columns in feature_sets.items():
            experiments.append((pt_label, pt_min, pt_max, feature_name, columns))

    print(f"Running {len(experiments)} experiments in parallel...")

    if n_jobs == 1:
        results = []
        for exp in experiments:
            result = run_single_experiment(*exp)
            if result:
                results.append(result)
                print(f"Completed: {exp[0]}-{exp[3]} "
                      f"(ARI: {result['ARI']:.3f}, Silhouette: {result['Silhouette']:.3f})")
    else:
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(run_single_experiment)(*exp) for exp in experiments
        )
        results = [r for r in results if r is not None]

    if results:
        results_df = pd.DataFrame(results)
        os.makedirs("kmeans_all", exist_ok=True)
        results_path = os.path.join("kmeans_all", f"clustering_results_optimized_{medium}.csv")
        results_df.to_csv(results_path, index=False)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETED!")
        print(f"Results saved at: {results_path}")
        print(f"Experiments executed: {len(results)}")
        print("="*60)

        print("\nTop 3 results by ARI:")
        top_ari = results_df.nlargest(3, 'ARI')[
            ['pt_range', 'feature_set', 'ARI', 'Silhouette', 'F1_Score']
        ]
        print(top_ari.to_string(index=False))

    else:
        print("No valid results obtained!")


if __name__ == "__main__":
    main()

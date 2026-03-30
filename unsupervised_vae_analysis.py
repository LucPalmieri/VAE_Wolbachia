"""
UNSUPERVISED VAE ANALYSIS
=========================
Trains VAE using two-stage approach and discovers structure in the latent
space WITHOUT using biological labels. Clustering is applied purely on the
learned latent representation.

Multiple clustering algorithms are tested across a range of k values:
  - K-Means
  - Agglomerative (hierarchical)
  - Gaussian Mixture Models (GMM)

The best solution is selected by highest silhouette score across all
algorithms and k values. Biological labels (if present in the data file)
are stored but NOT used during analysis — they are shown only in a
post-hoc comparison table to help interpret the results.

Usage:
    python unsupervised_vae_analysis.py <datafile> [options]

Examples:
    python unsupervised_vae_analysis.py wolbachia_47genes_host_newIDs.txt
    python unsupervised_vae_analysis.py wolbachia_47genes_strain_newIDs.txt --runs 10
    python unsupervised_vae_analysis.py wolbachia_47genes_country_newIDs.txt --latent-dim 2
    python unsupervised_vae_analysis.py wolbachia_47genes_host_newIDs.txt --k-min 2 --k-max 6

Output is saved to: unsupervised_results/
"""

import os
import sys
import json
import argparse
import re
import traceback
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras.layers import *
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.metrics import categorical_crossentropy
import tensorflow.compat.v1.keras.backend as K

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(
    description='Unsupervised VAE analysis — no biological labels used',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python unsupervised_vae_analysis.py wolbachia_47genes_host_newIDs.txt
    python unsupervised_vae_analysis.py wolbachia_47genes_strain_newIDs.txt --runs 10
    python unsupervised_vae_analysis.py wolbachia_47genes_country_newIDs.txt --k-min 2 --k-max 6

Output will be saved to: unsupervised_results/
    """
)
parser.add_argument('datafile', type=str,
                    help='Input data file (e.g., wolbachia_47genes_host_newIDs.txt)')
parser.add_argument('--runs', type=int, default=5,
                    help='Number of independent VAE runs (default: 5)')
parser.add_argument('--latent-dim', type=int, default=2,
                    help='Latent space dimensions (default: 2)')
parser.add_argument('--k-min', type=int, default=2,
                    help='Minimum number of clusters to test (default: 2)')
parser.add_argument('--k-max', type=int, default=5,
                    help='Maximum number of clusters to test (default: 5)')
args = parser.parse_args()

# ============================================================================
# EXTRACT CATEGORY FROM FILENAME AND SET UP DIRECTORIES
# ============================================================================
DATA_FILE = args.datafile

match = re.search(r'47genes_(.+?)_newIDs\.txt', DATA_FILE)
if match:
    category = match.group(1)
else:
    category = os.path.splitext(os.path.basename(DATA_FILE))[0]

OUTPUT_DIR = "unsupervised_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_RUNS      = args.runs
LATENT_DIM  = args.latent_dim
K_MIN       = args.k_min
K_MAX       = args.k_max

TRAINING_ROUNDS   = 4
KL_WEIGHT_STAGE1  = 0.3
KL_WEIGHT_STAGE2  = 1.0

print("=" * 70)
print("UNSUPERVISED VAE ANALYSIS")
print("=" * 70)
print(f"Started:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input file:       {DATA_FILE}")
print(f"Category:         {category}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Number of runs:   {N_RUNS}")
print(f"Latent dimensions:{LATENT_DIM}")
print(f"k range:          {K_MIN} – {K_MAX}")
print(f"Training:         Two-stage approach (~4000 epochs/run)")
print(f"  Stage 1:        KL weight = {KL_WEIGHT_STAGE1}")
print(f"  Stage 2:        KL weight = {KL_WEIGHT_STAGE2}")
print("NOTE: Biological labels loaded but NOT used in analysis.")
print("      They appear only in the post-hoc comparison.")
print("=" * 70 + "\n")

# ============================================================================
# HELPER: numpy → Python types for JSON
# ============================================================================
def convert_to_python_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    return obj

# ============================================================================
# LOAD DATA  (labels stored but not passed to any metric function)
# ============================================================================
print("Loading data...")

def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file '{filename}' not found!")
    tmp = {"data_set": filename.replace(".txt", ""),
           "name": [], "group": [], "one_hot": []}
    with open(filename, "r") as fh:
        for line in fh:
            tokens = line.rstrip("\n").split(" ")
            tmp["name"].append(tokens[0])
            tmp["group"].append(tokens[1])     # stored but not used
            tmp["one_hot"].append([])
            for token in tokens[2:]:
                vector = list(map(float, token.strip("[]").split(",")))
                tmp["one_hot"][-1].append(vector)
    tmp["name"]    = np.array(tmp["name"],    str)
    tmp["group"]   = np.array(tmp["group"],   str)
    tmp["one_hot"] = np.array(tmp["one_hot"], float)
    tmp["shape"]   = tmp["one_hot"].shape
    return tmp

try:
    data            = load_data(DATA_FILE)
    sample_names    = data["name"]
    biological_labels = data["group"]   # kept only for post-hoc table
    X               = data["one_hot"]

    print(f"  Samples:    {len(sample_names)}")
    print(f"  Data shape: {X.shape}  (samples, genes, categories)")
    print(f"  Biological labels in file (not used): {np.unique(biological_labels)}\n")

except Exception as e:
    print(f"ERROR loading data: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# UNSUPERVISED CLUSTERING  (no labels, pure internal metrics)
# ============================================================================
def run_clustering(latent_coords, k_min, k_max):
    """
    Test K-Means, Agglomerative, and GMM for every k in [k_min, k_max].
    Returns a list of result dicts sorted by silhouette score (best first).
    """
    results = []
    n_samples = latent_coords.shape[0]

    for k in range(k_min, k_max + 1):
        algorithms = {
            "KMeans":        KMeans(n_clusters=k, n_init=20, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=k),
            "GMM":           GaussianMixture(n_components=k, n_init=5, random_state=42),
        }
        for algo_name, algo in algorithms.items():
            try:
                if algo_name == "GMM":
                    labels_pred = algo.fit_predict(latent_coords)
                else:
                    labels_pred = algo.fit_predict(latent_coords)

                # Need at least 2 non-trivial clusters
                unique_clusters = np.unique(labels_pred)
                if len(unique_clusters) < 2:
                    continue

                sil = float(silhouette_score(latent_coords, labels_pred))
                db  = float(davies_bouldin_score(latent_coords, labels_pred))
                ch  = float(calinski_harabasz_score(latent_coords, labels_pred))

                # Separation ratio: between-centroid dist / sqrt(within-group var)
                centroids = np.array([latent_coords[labels_pred == c].mean(axis=0)
                                      for c in unique_clusters])
                avg_between = float(np.mean(pdist(centroids, 'euclidean')))
                within_vars = [np.mean(np.sum(
                    (latent_coords[labels_pred == c] - centroids[i]) ** 2, axis=1))
                    for i, c in enumerate(unique_clusters)]
                avg_within = float(np.mean(within_vars))
                sep_ratio  = avg_between / (np.sqrt(avg_within) + 1e-10)

                results.append({
                    "algorithm":       algo_name,
                    "k":               k,
                    "labels":          labels_pred.tolist(),
                    "silhouette":      sil,
                    "davies_bouldin":  db,
                    "calinski_harabasz": ch,
                    "separation_ratio": float(sep_ratio),
                    "avg_between_distance": avg_between,
                    "avg_within_variance":  avg_within,
                })

            except Exception as e:
                print(f"    Warning: {algo_name} k={k} failed: {e}")

    # Sort best first by silhouette
    results.sort(key=lambda r: r["silhouette"], reverse=True)
    return results

# ============================================================================
# PLOTTING HELPERS
# ============================================================================
def get_colors(n):
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 1, 10))[:n]
    elif n <= 20:
        return plt.cm.tab20(np.linspace(0, 1, 20))[:n]
    return plt.cm.rainbow(np.linspace(0, 1, n))


def plot_clusters(ax, z, cluster_labels, sample_names, title="Latent Space"):
    """Scatter plot coloured by discovered clusters (2D)."""
    unique_clusters = np.unique(cluster_labels)
    colors = get_colors(len(unique_clusters))
    for i, c in enumerate(unique_clusters):
        mask = cluster_labels == c
        ax.scatter(z[mask, 0], z[mask, 1],
                   c=[colors[i]], label=f"Cluster {c}",
                   s=120, alpha=0.8, edgecolors='black', linewidth=1.5)
    for i, name in enumerate(sample_names):
        ax.text(z[i, 0], z[i, 1], name, fontsize=6, ha='right', va='bottom', alpha=0.8)
    ax.set_xlabel('Latent Dimension 1', fontsize=11)
    ax.set_ylabel('Latent Dimension 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_clusters_uncertainty(ax, mu, sigma, cluster_labels, sample_names,
                               title="Latent Space with Uncertainty", n_samples=150):
    """Scatter plot with uncertainty clouds, coloured by discovered clusters (2D)."""
    unique_clusters = np.unique(cluster_labels)
    colors = get_colors(len(unique_clusters))
    # Uncertainty clouds
    for i, c in enumerate(unique_clusters):
        for idx in np.where(cluster_labels == c)[0]:
            samples = mu[idx] + np.random.normal(0, 1, size=(n_samples, LATENT_DIM)) * sigma[idx]
            ax.scatter(samples[:, 0], samples[:, 1],
                       c=[colors[i]], alpha=0.06, s=6, edgecolors='none')
    # Means
    for i, c in enumerate(unique_clusters):
        mask = cluster_labels == c
        ax.scatter(mu[mask, 0], mu[mask, 1],
                   c=[colors[i]], label=f"Cluster {c}",
                   s=120, alpha=1.0, edgecolors='black', linewidth=2)
    for i, name in enumerate(sample_names):
        ax.text(mu[i, 0], mu[i, 1], name, fontsize=6, ha='right', va='bottom', alpha=0.8)
    ax.set_xlabel('Latent Dimension 1', fontsize=11)
    ax.set_ylabel('Latent Dimension 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_silhouette_heatmap(ax, all_clustering_results):
    """
    Heatmap: rows = algorithms, columns = k values,
    cells = silhouette score (best combination highlighted).
    """
    algorithms = ["KMeans", "Agglomerative", "GMM"]
    k_values   = list(range(K_MIN, K_MAX + 1))

    grid = np.full((len(algorithms), len(k_values)), np.nan)
    for r in all_clustering_results:
        row = algorithms.index(r["algorithm"])
        col = k_values.index(r["k"])
        if np.isnan(grid[row, col]) or r["silhouette"] > grid[row, col]:
            grid[row, col] = r["silhouette"]

    im = ax.imshow(grid, cmap='RdYlGn', vmin=-0.1, vmax=0.7, aspect='auto')
    plt.colorbar(im, ax=ax, label='Silhouette Score')
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"k={k}" for k in k_values])
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    ax.set_title('Silhouette Score by Algorithm × k', fontweight='bold')

    # Annotate cells and mark best
    best_val = np.nanmax(grid)
    for i in range(len(algorithms)):
        for j in range(len(k_values)):
            if not np.isnan(grid[i, j]):
                marker = "★" if grid[i, j] == best_val else ""
                ax.text(j, i, f"{grid[i,j]:.3f}{marker}",
                        ha='center', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# VAE MODEL
# ============================================================================
def build_two_stage_vae(original_dim, cat, latent_dim=2):
    def act_fn(fn, tensor):
        return LeakyReLU()(tensor) if fn == "leakyrelu" else Activation(fn)(tensor)

    half_cat = int(cat / 2)
    x_in     = Input(shape=(original_dim, cat), name="x_in")
    x_in_em  = Dense(half_cat, use_bias=False, name="x_in_em")(x_in)

    en = Flatten()(x_in_em)
    en = BatchNormalization(scale=False, center=False)(en)
    for dim, drop in zip([200, 200, 200], [0.2, 0.2, 0.2]):
        en = Dense(dim)(en)
        en = Dropout(drop)(en)
        en = act_fn("elu", en)
        en = BatchNormalization(scale=False, center=False)(en)

    Z_mu          = Dense(latent_dim)(en)
    Z_log_sigma_sq = Dense(latent_dim)(en)
    Z_sigma       = Lambda(lambda x: K.exp(0.5 * x))(Z_log_sigma_sq)
    Z             = Lambda(lambda x: x[0] + x[1] * K.random_normal(K.shape(x[0])))([Z_mu, Z_sigma])

    de = Z
    for dim, drop in zip([200, 200, 200], [0.2, 0.2, 0.2]):
        de = Dense(dim)(de)
        de = Dropout(drop)(de)
        de = act_fn("elu", de)
        de = BatchNormalization(scale=False, center=False)(de)

    de      = Dense(original_dim * half_cat)(de)
    x_out_em = Reshape((-1, half_cat))(de)
    x_out   = Dense(cat, activation="softmax")(x_out_em)

    def vae_loss(kl_weight=1.0):
        def loss(x_true, x_pred):
            mask    = K.sum(x_in, axis=-1)
            kl_loss = 0.5 * K.sum(K.square(Z_mu) + K.square(Z_sigma)
                                   - 2.0 * K.log(Z_sigma) - 1.0, axis=-1)
            recon   = K.sum(categorical_crossentropy(x_in, x_out) * mask, axis=-1)
            return K.mean(recon + kl_loss * kl_weight)
        return loss

    def acc(x_true, x_pred):
        mask    = K.sum(x_in, axis=-1, keepdims=True)
        acc_val = K.sum(K.square(x_in - x_out), axis=-1, keepdims=True)
        return K.mean(1.0 - K.sqrt(K.sum(acc_val * mask, axis=1) / K.sum(mask, axis=1)))

    vae1 = Model([x_in], [x_out], name="vae_stage1")
    vae1.compile(optimizer='adam', loss=vae_loss(KL_WEIGHT_STAGE1), metrics=[acc])

    vae2 = Model([x_in], [x_out], name="vae_stage2")
    vae2.compile(optimizer='adam', loss=vae_loss(KL_WEIGHT_STAGE2), metrics=[acc])

    enc = Model([x_in], [Z_mu, Z_sigma], name="encoder")
    return vae1, vae2, enc


def data_generator(data, batch_size):
    while True:
        idx = np.random.randint(0, data.shape[0], size=batch_size)
        tmp = data[idx]
        yield tmp, tmp


def train_two_stage_vae(data, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    K.clear_session()

    vae1, vae2, enc = build_two_stage_vae(data.shape[1], data.shape[2], LATENT_DIM)
    loss_history, acc_history = [], []

    print(f"    Stage 1 (KL={KL_WEIGHT_STAGE1})...")
    for i in range(TRAINING_ROUNDS):
        f          = 1.0 / (TRAINING_ROUNDS - i)
        batch_size = int(data.shape[0] * f + 0.5)
        steps      = int(data.shape[0] / batch_size + 0.5)
        epochs     = int(1000 * f + 0.5)
        history    = vae1.fit_generator(data_generator(data, batch_size),
                                        steps_per_epoch=steps, epochs=epochs, verbose=0)
        loss_history += list(history.history['loss'])
        acc_history  += list(history.history['acc'])

    stage1_end = len(loss_history)
    vae2.set_weights(vae1.get_weights())

    print(f"    Stage 2 (KL={KL_WEIGHT_STAGE2})...")
    for i in range(TRAINING_ROUNDS):
        f          = 1.0 / (TRAINING_ROUNDS - i)
        batch_size = int(data.shape[0] * f + 0.5)
        steps      = int(data.shape[0] / batch_size + 0.5)
        epochs     = int(1000 * f + 0.5)
        history    = vae2.fit_generator(data_generator(data, batch_size),
                                        steps_per_epoch=steps, epochs=epochs, verbose=0)
        loss_history += list(history.history['loss'])
        acc_history  += list(history.history['acc'])

    vae_mu, vae_sigma = enc.predict(data)
    return vae_mu, vae_sigma, loss_history, acc_history, stage1_end

# ============================================================================
# RUN INDEPENDENT VAE TRAININGS
# ============================================================================
all_run_results = []

for run in range(1, N_RUNS + 1):
    print(f"{'='*70}")
    print(f"RUN {run}/{N_RUNS}")
    print(f"{'='*70}")

    run_seed = 42 + run

    try:
        print(f"  Training VAE (seed={run_seed})...")
        Z_mu_pred, Z_sigma_pred, loss_hist, acc_hist, stage1_end = \
            train_two_stage_vae(X, run_seed)

        total_epochs = len(loss_hist)
        final_loss   = float(loss_hist[-1])
        print(f"  Training complete! Epochs: {total_epochs}, Final loss: {final_loss:.4f}")

        # ── Unsupervised clustering on latent space ──────────────────────────
        print(f"  Running clustering (k={K_MIN}–{K_MAX})...")
        clustering_results = run_clustering(Z_mu_pred, K_MIN, K_MAX)

        if not clustering_results:
            print("  Warning: no valid clustering solution found for this run.")
            continue

        best_cluster = clustering_results[0]
        print(f"  Best solution: {best_cluster['algorithm']} k={best_cluster['k']}"
              f"  Silhouette={best_cluster['silhouette']:.4f}")

        run_result = {
            'run':              int(run),
            'seed':             int(run_seed),
            'total_epochs':     int(total_epochs),
            'final_loss':       final_loss,
            'best_clustering':  best_cluster,
            'all_clustering':   clustering_results,
            'latent_mu':        Z_mu_pred.tolist(),
            'latent_sigma':     Z_sigma_pred.tolist(),
            'loss_history':     [float(x) for x in loss_hist],
            'acc_history':      [float(x) for x in acc_hist],
            'stage1_end':       int(stage1_end),
        }
        all_run_results.append(run_result)
        print(f"Run {run} complete!\n")
        K.clear_session()

    except Exception as e:
        print(f"\nERROR in run {run}: {str(e)}")
        traceback.print_exc()
        print()
        continue

# ============================================================================
# AGGREGATE RESULTS
# ============================================================================
print("=" * 70)
print("AGGREGATING RESULTS")
print("=" * 70)

if not all_run_results:
    print("ERROR: No successful runs!")
    sys.exit(1)

silhouettes = [r['best_clustering']['silhouette']  for r in all_run_results]
dbs         = [r['best_clustering']['davies_bouldin'] for r in all_run_results]
chs         = [r['best_clustering']['calinski_harabasz'] for r in all_run_results]
seps        = [r['best_clustering']['separation_ratio']  for r in all_run_results]

sil_mean, sil_std = float(np.mean(silhouettes)), float(np.std(silhouettes))
sil_cv            = float((sil_std / sil_mean * 100) if sil_mean > 0 else 0)
db_mean,  db_std  = float(np.mean(dbs)),  float(np.std(dbs))
ch_mean,  ch_std  = float(np.mean(chs)),  float(np.std(chs))
sep_mean, sep_std = float(np.mean(seps)), float(np.std(seps))

best_run = max(all_run_results, key=lambda r: r['best_clustering']['silhouette'])

print(f"\nSuccessful runs: {len(all_run_results)}/{N_RUNS}")
print(f"\nBest clustering solution per run:")
for r in all_run_results:
    bc = r['best_clustering']
    print(f"  Run {r['run']:2d}: {bc['algorithm']:15s} k={bc['k']}  "
          f"Sil={bc['silhouette']:.4f}  DB={bc['davies_bouldin']:.4f}")

print(f"\nSilhouette Score:  {sil_mean:.4f} ± {sil_std:.4f}  (CV={sil_cv:.2f}%)")
print(f"Davies-Bouldin:    {db_mean:.4f}  ± {db_std:.4f}")
print(f"Calinski-Harabasz: {ch_mean:.2f}  ± {ch_std:.2f}")
print(f"Separation Ratio:  {sep_mean:.4f} ± {sep_std:.4f}")
print(f"\nBEST RUN: #{best_run['run']}  "
      f"{best_run['best_clustering']['algorithm']} k={best_run['best_clustering']['k']}  "
      f"Silhouette={best_run['best_clustering']['silhouette']:.4f}")

# ============================================================================
# POST-HOC COMPARISON  (discovered clusters vs. biological labels)
# ============================================================================
best_cluster_labels = np.array(best_run['best_clustering']['labels'])
best_k              = best_run['best_clustering']['k']

print("\n" + "=" * 70)
print("POST-HOC COMPARISON: Discovered clusters vs. biological labels")
print("(Labels were NOT used during analysis — this is purely informational)")
print("=" * 70)

bio_unique = np.unique(biological_labels)
cluster_ids = np.unique(best_cluster_labels)

# Print cross-tabulation
header = f"{'Biological label':20s}" + "".join([f"  Cluster {c}" for c in cluster_ids])
print(header)
print("-" * len(header))
for bio in bio_unique:
    row = f"{bio:20s}"
    for c in cluster_ids:
        count = int(np.sum((biological_labels == bio) & (best_cluster_labels == c)))
        row += f"  {count:9d}"
    print(row)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

best_mu    = np.array(best_run['latent_mu'])
best_sigma = np.array(best_run['latent_sigma'])

# Text file: one row per sample, latent coords + discovered cluster
header_cols = ["sample", "discovered_cluster", "biological_label_posthoc"]
header_cols += [f"latent_dim{d+1}" for d in range(LATENT_DIM)]
header_cols += ["silhouette_score", "convergence_cv"]

output_lines = ["\t".join(header_cols)]
for i, name in enumerate(sample_names):
    row = [name,
           str(best_cluster_labels[i]),
           biological_labels[i]]
    row += [f"{best_mu[i, d]:.6f}" for d in range(LATENT_DIM)]
    row += [f"{best_run['best_clustering']['silhouette']:.4f}",
            f"{sil_cv:.2f}"]
    output_lines.append("\t".join(row))

output_file = os.path.join(OUTPUT_DIR, "unsupervised_latent_coordinates.txt")
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"  Latent coordinates saved to: {output_file}")

# JSON
all_run_results_clean = convert_to_python_types(all_run_results)
results_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'configuration': {
        'approach':          'unsupervised',
        'n_runs':            int(N_RUNS),
        'latent_dim':        int(LATENT_DIM),
        'k_min':             int(K_MIN),
        'k_max':             int(K_MAX),
        'two_stage_training': True,
        'kl_weight_stage1':  float(KL_WEIGHT_STAGE1),
        'kl_weight_stage2':  float(KL_WEIGHT_STAGE2),
    },
    'convergence': {
        'successful_runs':       int(len(all_run_results)),
        'silhouette_mean':       sil_mean,
        'silhouette_std':        sil_std,
        'silhouette_cv_percent': sil_cv,
        'davies_bouldin_mean':   db_mean,
        'calinski_harabasz_mean': ch_mean,
        'separation_ratio_mean': sep_mean,
    },
    'best_run': {
        'run_number':   int(best_run['run']),
        'algorithm':    best_run['best_clustering']['algorithm'],
        'k':            best_run['best_clustering']['k'],
        'metrics':      best_run['best_clustering'],
        'total_epochs': best_run['total_epochs'],
    },
    'all_runs': all_run_results_clean,
}

json_file = os.path.join(OUTPUT_DIR, "unsupervised_results_summary.json")
with open(json_file, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"  JSON results saved to: {json_file}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nCreating comprehensive visualization...")

fig = plt.figure(figsize=(22, 14))

best_loss    = best_run['loss_history']
best_acc     = best_run['acc_history']
stage1_end   = best_run['stage1_end']

# 1 — Training loss
ax1 = plt.subplot(2, 4, 1)
ax1.plot(best_loss, linewidth=1, color='steelblue')
ax1.axvline(x=stage1_end, color='red', linestyle='--', linewidth=2, label='Stage 2 Start')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title(f'Training Loss (Best Run #{best_run["run"]})', fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3)

# 2 — Training accuracy
ax2 = plt.subplot(2, 4, 2)
ax2.plot(best_acc, linewidth=1, color='seagreen')
ax2.axvline(x=stage1_end, color='red', linestyle='--', linewidth=2, label='Stage 2 Start')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title(f'Training Accuracy (Best Run #{best_run["run"]})', fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)

# 3 — Convergence across runs
ax3 = plt.subplot(2, 4, 3)
runs_x = [r['run'] for r in all_run_results]
sils   = [r['best_clustering']['silhouette'] for r in all_run_results]
ax3.plot(runs_x, sils, 'o-', linewidth=2, markersize=10, color='steelblue')
ax3.axhline(y=sil_mean, color='red', linestyle='--', linewidth=2, label=f'Mean={sil_mean:.3f}')
ax3.fill_between(runs_x,
                 [sil_mean - sil_std] * len(runs_x),
                 [sil_mean + sil_std] * len(runs_x),
                 alpha=0.2, color='red')
ax3.set_xlabel('Run'); ax3.set_ylabel('Best Silhouette')
ax3.set_title(f'Convergence (CV={sil_cv:.1f}%)', fontweight='bold')
ax3.legend(); ax3.grid(True, alpha=0.3)

# 4 — Silhouette heatmap (algorithm × k) from best run
ax4 = plt.subplot(2, 4, 4)
plot_silhouette_heatmap(ax4, best_run['all_clustering'])

# 5 — Latent space coloured by discovered clusters
ax5 = plt.subplot(2, 4, 5)
plot_clusters(ax5, best_mu, best_cluster_labels, sample_names,
              title=f'Discovered Clusters ({best_run["best_clustering"]["algorithm"]} k={best_k})')

# 6 — Latent space with uncertainty
ax6 = plt.subplot(2, 4, 6)
plot_clusters_uncertainty(ax6, best_mu, best_sigma, best_cluster_labels, sample_names,
                          title='Discovered Clusters with Uncertainty')

# 7 — Latent space coloured by biological labels (post-hoc only)
ax7 = plt.subplot(2, 4, 7)
bio_colors = get_colors(len(bio_unique))
for i, bio in enumerate(bio_unique):
    mask = biological_labels == bio
    ax7.scatter(best_mu[mask, 0], best_mu[mask, 1],
                c=[bio_colors[i]], label=bio.upper(),
                s=120, alpha=0.8, edgecolors='black', linewidth=1.5)
for i, name in enumerate(sample_names):
    ax7.text(best_mu[i, 0], best_mu[i, 1], name, fontsize=6, ha='right', va='bottom', alpha=0.8)
ax7.set_xlabel('Latent Dimension 1'); ax7.set_ylabel('Latent Dimension 2')
ax7.set_title('Post-hoc: Biological Labels (NOT used in analysis)', fontsize=10, fontweight='bold')
ax7.legend(loc='best', fontsize=9); ax7.grid(True, alpha=0.3)

# 8 — Summary text
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
bc = best_run['best_clustering']
summary_text = f"""
UNSUPERVISED VAE ANALYSIS
{'='*38}

APPROACH: No biological labels used
  Clustering: {bc['algorithm']}
  Best k: {bc['k']}

RUNS: {len(all_run_results)}/{N_RUNS} successful

TWO-STAGE TRAINING:
  Stage 1: KL = {KL_WEIGHT_STAGE1}
  Stage 2: KL = {KL_WEIGHT_STAGE2}
  Total epochs: ~{best_run['total_epochs']}

SEPARATION METRICS (best run):
  Silhouette:  {bc['silhouette']:.4f}
  Davies-Bouldin: {bc['davies_bouldin']:.4f}
  Calinski-H:  {bc['calinski_harabasz']:.2f}
  Sep. Ratio:  {bc['separation_ratio']:.4f}

CONVERGENCE:
  Mean Sil: {sil_mean:.4f} ± {sil_std:.4f}
  CV: {sil_cv:.2f}%  {'EXCELLENT' if sil_cv < 5 else 'GOOD' if sil_cv < 10 else 'REVIEW'}

{'='*38}
"""
ax8.text(0.05, 0.5, summary_text, fontsize=8.5, verticalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

plt.suptitle(
    f'Unsupervised VAE Analysis — {DATA_FILE}\n'
    f'Best solution: {bc["algorithm"]} k={bc["k"]} | '
    f'Silhouette={bc["silhouette"]:.4f} | CV={sil_cv:.2f}%',
    fontsize=15, fontweight='bold')
plt.tight_layout()

plot_file = os.path.join(OUTPUT_DIR, "unsupervised_analysis_summary.pdf")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"  Comprehensive plot saved to: {plot_file}")
plt.close()

# ============================================================================
# PUBLICATION-QUALITY FIGURE
# ============================================================================
print("\nCreating publication-quality figure...")

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

# Left: discovered clusters
plot_clusters(axes[0], best_mu, best_cluster_labels, sample_names,
              title=f'Discovered Clusters\n({bc["algorithm"]} k={bc["k"]})')

# Centre: with uncertainty
plot_clusters_uncertainty(axes[1], best_mu, best_sigma, best_cluster_labels, sample_names,
                          title='Discovered Clusters with Uncertainty')

# Right: biological labels (post-hoc)
for i, bio in enumerate(bio_unique):
    mask = biological_labels == bio
    axes[2].scatter(best_mu[mask, 0], best_mu[mask, 1],
                    c=[bio_colors[i]], label=bio.upper(),
                    s=120, alpha=0.8, edgecolors='black', linewidth=1.5)
for i, name in enumerate(sample_names):
    axes[2].text(best_mu[i, 0], best_mu[i, 1], name, fontsize=6,
                 ha='right', va='bottom', alpha=0.8)
axes[2].set_xlabel('Latent Dimension 1'); axes[2].set_ylabel('Latent Dimension 2')
axes[2].set_title('Post-hoc: Biological Labels\n(NOT used in analysis)', fontweight='bold')
axes[2].legend(loc='best', fontsize=9); axes[2].grid(True, alpha=0.3)

plt.suptitle(
    f'Wolbachia Unsupervised VAE — Latent Space Structure\n'
    f'Silhouette: {bc["silhouette"]:.4f} | Convergence CV: {sil_cv:.2f}% | '
    f'Sep. Ratio: {bc["separation_ratio"]:.2f}',
    fontsize=13, fontweight='bold')
plt.tight_layout()

pub_file = os.path.join(OUTPUT_DIR, "publication_figure_unsupervised.pdf")
plt.savefig(pub_file, dpi=300, bbox_inches='tight')
print(f"  Publication figure saved to: {pub_file}")
plt.close()

# ============================================================================
# FINAL REPORT
# ============================================================================
report_file = os.path.join(OUTPUT_DIR, "UNSUPERVISED_ANALYSIS_REPORT.txt")
with open(report_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("UNSUPERVISED VAE ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("APPROACH:\n")
    f.write("  Fully unsupervised — biological labels NOT used during analysis.\n")
    f.write("  The VAE learns a latent representation; clustering algorithms\n")
    f.write("  then discover structure without any prior grouping information.\n\n")

    f.write("CONFIGURATION:\n")
    f.write(f"  Number of runs:    {N_RUNS}\n")
    f.write(f"  Successful runs:   {len(all_run_results)}\n")
    f.write(f"  Latent dimensions: {LATENT_DIM}\n")
    f.write(f"  k range tested:    {K_MIN} – {K_MAX}\n")
    f.write(f"  Algorithms:        KMeans, Agglomerative, GMM\n")
    f.write(f"  Stage 1 KL:        {KL_WEIGHT_STAGE1}\n")
    f.write(f"  Stage 2 KL:        {KL_WEIGHT_STAGE2}\n\n")

    f.write("=" * 70 + "\n")
    f.write("BEST SOLUTION:\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"  Algorithm:           {bc['algorithm']}\n")
    f.write(f"  k (clusters):        {bc['k']}\n")
    f.write(f"  Silhouette Score:    {bc['silhouette']:.4f}\n")
    f.write(f"  Davies-Bouldin:      {bc['davies_bouldin']:.4f}\n")
    f.write(f"  Calinski-Harabasz:   {bc['calinski_harabasz']:.2f}\n")
    f.write(f"  Separation Ratio:    {bc['separation_ratio']:.4f}\n\n")

    f.write("=" * 70 + "\n")
    f.write("CONVERGENCE (across all runs):\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"  Silhouette Mean:  {sil_mean:.4f} ± {sil_std:.4f}\n")
    f.write(f"  CV:               {sil_cv:.2f}%\n")
    if sil_cv < 5:
        f.write("EXCELLENT — CV < 5% (highly reproducible)\n\n")
    elif sil_cv < 10:
        f.write("GOOD — CV 5–10% (reproducible)\n\n")
    else:
        f.write("REVIEW — CV > 10% (variable across runs)\n\n")

    f.write("ALL RUNS:\n")
    for r in all_run_results:
        bc_r = r['best_clustering']
        f.write(f"  Run {r['run']:2d}: {bc_r['algorithm']:15s} k={bc_r['k']}  "
                f"Sil={bc_r['silhouette']:.4f}  DB={bc_r['davies_bouldin']:.4f}  "
                f"Epochs={r['total_epochs']}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("POST-HOC COMPARISON: Discovered Clusters vs. Biological Labels\n")
    f.write("(Biological labels were NOT used during the analysis)\n")
    f.write("=" * 70 + "\n\n")
    header = f"{'Biological label':20s}" + \
             "".join([f"  Cluster {c}" for c in np.unique(best_cluster_labels)])
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for bio in bio_unique:
        row = f"{bio:20s}"
        for c in np.unique(best_cluster_labels):
            count = int(np.sum((biological_labels == bio) & (best_cluster_labels == c)))
            row += f"  {count:9d}"
        f.write(row + "\n")
    f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("OUTPUT FILES:\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"1. {output_file}\n   Latent coordinates + discovered cluster per sample\n\n")
    f.write(f"2. {json_file}\n   Complete results in JSON format\n\n")
    f.write(f"3. {plot_file}\n   Comprehensive analysis dashboard\n\n")
    f.write(f"4. {pub_file}\n   Publication-quality figure (300 DPI)\n\n")
    f.write("=" * 70 + "\n")

print(f"  Report saved to: {report_file}")

print("\n" + "=" * 70)
print("UNSUPERVISED ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nBest solution: {bc['algorithm']}  k={bc['k']}")
print(f"  Silhouette Score:    {bc['silhouette']:.4f}")
print(f"  Separation Ratio:    {bc['separation_ratio']:.4f}")
print(f"  Convergence CV:      {sil_cv:.2f}%")
print(f"\nResults saved to: {OUTPUT_DIR}/")
print(f"{report_file}")
print(f"{pub_file}")
print("=" * 70)
###
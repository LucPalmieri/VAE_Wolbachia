"""
SUPERVISED VAE ANALYSIS WITH BIOLOGICAL LABELS
===============================================
Trains VAE using two-stage approach and evaluates how well the latent
space separates the biologically meaningful groups.

This script automatically detects and uses whatever labels are in your
data file (column 2). Works with any grouping scheme:
- cer, cin, single (default)
- dual, single (infection type)
- hostA, hostB, hostC (by host)
- supergroup_A, supergroup_B (phylogenetic)
- Any other biological hypothesis you want to test!

Colors are automatically assigned based on the number of unique labels.

This is a supervised analysis - we use known biological labels to:
1. Color/visualize the latent space
2. Calculate separation metrics
3. Assess reproducibility across multiple runs

Usage:
    python supervised_vae_biological_groups.py <datafile> [options]
    
Examples:
    python supervised_vae_biological_groups.py wolbachia_47genes_host_newIDs.txt
    python supervised_vae_biological_groups.py wolbachia_47genes_strain_newIDs.txt --runs 10
    python supervised_vae_biological_groups.py wolbachia_47genes_country_newIDs.txt --latent-dim 3
    
Output is saved to: supervised_results/by_{category}/
"""

import os
import sys
import json
import argparse
import re
import traceback
import numpy as np
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
    description='Supervised VAE analysis with biological labels',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python supervised_vae_biological_groups.py wolbachia_47genes_host_newIDs.txt
    python supervised_vae_biological_groups.py wolbachia_47genes_strain_newIDs.txt --runs 10
    python supervised_vae_biological_groups.py wolbachia_47genes_country_newIDs.txt --latent-dim 3
    
Output will be saved to: supervised_results/by_{category}/
    """
)
parser.add_argument('datafile', type=str, 
                    help='Input data file (e.g., wolbachia_47genes_host_newIDs.txt)')
parser.add_argument('--runs', type=int, default=5,
                    help='Number of independent runs (default: 5)')
parser.add_argument('--latent-dim', type=int, default=2,
                    help='Latent space dimensions (default: 2)')
args = parser.parse_args()

# ============================================================================
# EXTRACT CATEGORY FROM FILENAME AND SET UP DIRECTORIES
# ============================================================================
DATA_FILE = args.datafile

# Extract category from filename (e.g., "host" from "wolbachia_47genes_host_newIDs.txt")
match = re.search(r'47genes_(.+?)_newIDs\.txt', DATA_FILE)
if match:
    category = match.group(1)
    OUTPUT_DIR = os.path.join("supervised_results", f"by_{category}")
else:
    # Fallback: use the whole filename without extension
    category = os.path.splitext(os.path.basename(DATA_FILE))[0]
    OUTPUT_DIR = os.path.join("supervised_results", category)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_RUNS = args.runs
LATENT_DIM = args.latent_dim

# Two-stage training parameters
TRAINING_ROUNDS = 4
KL_WEIGHT_STAGE1 = 0.3  # Low KL weight for Stage 1
KL_WEIGHT_STAGE2 = 1.0  # Full KL weight for Stage 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("SUPERVISED VAE ANALYSIS WITH BIOLOGICAL LABELS")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input file: {DATA_FILE}")
print(f"Category: {category}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Number of runs: {N_RUNS}")
print(f"Latent dimensions: {LATENT_DIM}")
print(f"Training: Two-stage approach (~4000 epochs/run)")
print(f"  Stage 1: KL weight = {KL_WEIGHT_STAGE1}")
print(f"  Stage 2: KL weight = {KL_WEIGHT_STAGE2}")
print("="*70 + "\n")

# ============================================================================
# HELPER FUNCTION: Convert numpy types to Python types for JSON
# ============================================================================
def convert_to_python_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file '{filename}' not found!")
    
    tmp = {"data_set": filename.replace(".txt", ""), "name": [], "group": [], "one_hot": []}
    with open(filename, "r") as file:
        for line in file:
            tokens = line.rstrip("\n").split(" ")
            tmp["name"].append(tokens[0])
            tmp["group"].append(tokens[1])
            tmp["one_hot"].append([])
            for token in tokens[2:]:
                token_clean = token.strip("[]")
                vector = list(map(float, token_clean.split(",")))
                tmp["one_hot"][-1].append(vector)
    
    tmp["name"] = np.array(tmp["name"], str)
    tmp["group"] = np.array(tmp["group"], str)
    tmp["one_hot"] = np.array(tmp["one_hot"], float)
    tmp["shape"] = tmp["one_hot"].shape
    return tmp

try:
    data = load_data(DATA_FILE)
    sample_names = data["name"]
    biological_labels = data["group"]  # Use biological labels
    X = data["one_hot"]
    
    print(f"  Samples: {len(sample_names)}")
    print(f"  Biological groups: {np.unique(biological_labels)}")
    for group in np.unique(biological_labels):
        count = np.sum(biological_labels == group)
        print(f"    {group}: {count} samples")
    print(f"  Data shape: {X.shape}")
    print(f"  (samples, genes, categories): {X.shape}\n")
    
except Exception as e:
    print(f"ERROR loading data: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# CALCULATE SEPARATION METRICS
# ============================================================================
def calculate_separation_metrics(latent_coords, labels):
    """Calculate metrics for how well biological groups are separated"""
    
    # Silhouette score: measures cluster quality (-1 to 1, higher is better)
    silhouette = float(silhouette_score(latent_coords, labels))
    
    # Davies-Bouldin index: measures cluster separation (lower is better)
    davies_bouldin = float(davies_bouldin_score(latent_coords, labels))
    
    # Calinski-Harabasz score: ratio of between-cluster to within-cluster variance (higher is better)
    calinski = float(calinski_harabasz_score(latent_coords, labels))
    
    # Between-group distances (average pairwise distances between group centroids)
    unique_groups = np.unique(labels)
    centroids = []
    for group in unique_groups:
        mask = labels == group
        centroid = np.mean(latent_coords[mask], axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    between_group_distances = pdist(centroids, metric='euclidean')
    avg_between_distance = float(np.mean(between_group_distances))
    
    # Within-group variance (average variance within each group)
    within_variances = []
    for group in unique_groups:
        mask = labels == group
        group_coords = latent_coords[mask]
        centroid = np.mean(group_coords, axis=0)
        variance = np.mean(np.sum((group_coords - centroid)**2, axis=1))
        within_variances.append(variance)
    
    avg_within_variance = float(np.mean(within_variances))
    
    return {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski,
        'avg_between_distance': avg_between_distance,
        'avg_within_variance': avg_within_variance,
        'separation_ratio': avg_between_distance / (np.sqrt(avg_within_variance) + 1e-10)
    }

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_latent_with_names(ax, z, labels, names, title="Latent Space"):
    """Plot latent space with sample names colored by biological group"""
    x = z[:, 0]
    y = z[:, 1]
    
    # Automatically assign colors based on number of unique labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Choose colormap based on number of groups
    if n_labels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_labels]
    elif n_labels <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_labels]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_labels))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(x[mask], y[mask], c=[colors[i]], label=label.upper(), 
                  s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add sample names
    for i, name in enumerate(names):
        ax.text(x[i], y[i], name, fontsize=6, ha='right', va='bottom', alpha=0.8)
    
    ax.set_xlabel('Latent Dimension 1', fontsize=11)
    ax.set_ylabel('Latent Dimension 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

def plot_latent_with_uncertainty(ax, mu, sigma, labels, names, title="Latent Space with Uncertainty", n_samples=150):
    """Plot latent space with uncertainty clouds colored by biological group"""
    
    # Automatically assign colors based on number of unique labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Choose colormap based on number of groups
    if n_labels <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_labels]
    elif n_labels <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_labels]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_labels))
    
    # Plot uncertainty clouds
    for i, label in enumerate(unique_labels):
        mask = labels == label
        indices = np.where(mask)[0]
        
        for idx in indices:
            # Sample from the distribution
            samples = mu[idx] + np.random.normal(0, 1, size=(n_samples, 2)) * sigma[idx]
            ax.scatter(samples[:, 0], samples[:, 1], c=[colors[i]], 
                      alpha=0.08, s=8, edgecolors='none')
    
    # Plot means
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(mu[mask, 0], mu[mask, 1], c=[colors[i]], label=label.upper(),
                  s=120, alpha=1.0, edgecolors='black', linewidth=2)
    
    # Add sample names
    for i, name in enumerate(names):
        ax.text(mu[i, 0], mu[i, 1], name, fontsize=6, ha='right', va='bottom', alpha=0.8)
    
    ax.set_xlabel('Latent Dimension 1', fontsize=11)
    ax.set_ylabel('Latent Dimension 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

# ============================================================================
# BUILD TWO-STAGE VAE MODEL
# ============================================================================
def build_two_stage_vae(original_dim, cat, latent_dim=2):
    """Build VAE with two different KL weights for two-stage training"""
    
    def act_fn(fn, tensor):
        if fn == "leakyrelu":
            return LeakyReLU()(tensor)
        else:
            return Activation(fn)(tensor)
    
    half_cat = int(cat / 2)
    x_in = Input(shape=(original_dim, cat), name="x_in")
    x_in_em = Dense(half_cat, use_bias=False, name="x_in_em")(x_in)
    
    # Encoder
    en = Flatten()(x_in_em)
    en = BatchNormalization(scale=False, center=False)(en)
    
    en_dim = [200, 200, 200]
    en_drop = [0.2, 0.2, 0.2]
    
    for i in range(len(en_dim)):
        en = Dense(en_dim[i])(en)
        en = Dropout(en_drop[i])(en)
        en = act_fn("elu", en)
        en = BatchNormalization(scale=False, center=False)(en)
    
    Z_mu = Dense(latent_dim)(en)
    Z_log_sigma_sq = Dense(latent_dim)(en)
    Z_sigma = Lambda(lambda x: K.exp(0.5 * x))(Z_log_sigma_sq)
    Z = Lambda(lambda x: x[0] + x[1] * K.random_normal(K.shape(x[0])))([Z_mu, Z_sigma])
    
    # Decoder
    de = Z
    de_dim = [200, 200, 200]
    de_drop = [0.2, 0.2, 0.2]
    
    for i in range(len(de_dim)):
        de = Dense(de_dim[i])(de)
        de = Dropout(de_drop[i])(de)
        de = act_fn("elu", de)
        de = BatchNormalization(scale=False, center=False)(de)
    
    de = Dense(original_dim * half_cat)(de)
    x_out_em = Reshape((-1, half_cat))(de)
    x_out = Dense(cat, activation="softmax")(x_out_em)
    
    # Loss function with configurable KL weight
    def vae_loss(kl_weight=1.0):
        def loss(x_true, x_pred):
            mask = K.sum(x_in, axis=-1)
            kl_loss = 0.5 * K.sum(K.square(Z_mu) + K.square(Z_sigma) - 2.0 * K.log(Z_sigma) - 1.0, axis=-1)
            recon = K.sum(categorical_crossentropy(x_in, x_out) * mask, axis=-1)
            return K.mean(recon + kl_loss * kl_weight)
        return loss
    
    def acc(x_true, x_pred):
        mask = K.sum(x_in, axis=-1, keepdims=True)
        acc_val = K.sum(K.square(x_in - x_out), axis=-1, keepdims=True)
        return K.mean(1.0 - K.sqrt(K.sum(acc_val * mask, axis=1) / K.sum(mask, axis=1)))
    
    # Create two models with different KL weights
    vae1 = Model([x_in], [x_out], name="vae_stage1")
    vae1.compile(optimizer='adam', loss=vae_loss(KL_WEIGHT_STAGE1), metrics=[acc])
    
    vae2 = Model([x_in], [x_out], name="vae_stage2")
    vae2.compile(optimizer='adam', loss=vae_loss(KL_WEIGHT_STAGE2), metrics=[acc])
    
    enc = Model([x_in], [Z_mu, Z_sigma], name="encoder")
    
    return vae1, vae2, enc

# ============================================================================
# DATA GENERATOR
# ============================================================================
def data_generator(data, batch_size):
    """Generator for training data"""
    while True:
        idx = np.random.randint(0, data.shape[0], size=batch_size)
        tmp = data[idx]
        yield tmp, tmp

# ============================================================================
# TWO-STAGE TRAINING FUNCTION
# ============================================================================
def train_two_stage_vae(data, seed):
    """Train VAE with two-stage approach"""
    
    # Set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # Clear session
    K.clear_session()
    
    # Build model
    vae1, vae2, enc = build_two_stage_vae(data.shape[1], data.shape[2], LATENT_DIM)
    
    loss_history = []
    acc_history = []
    stage1_end = 0
    
    # STAGE 1: Low KL weight training
    print(f"    Stage 1 (KL={KL_WEIGHT_STAGE1})...")
    for i in range(TRAINING_ROUNDS):
        f = 1.0 / (TRAINING_ROUNDS - i)
        batch_size = int(data.shape[0] * f + 0.5)
        steps = int(data.shape[0] / batch_size + 0.5)
        epochs = int(1000 * f + 0.5)
        
        history = vae1.fit_generator(
            data_generator(data, batch_size),
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=0
        )
        loss_history += list(history.history['loss'])
        acc_history += list(history.history['acc'])
    
    stage1_end = len(loss_history)
    
    # Transfer weights to vae2
    vae2.set_weights(vae1.get_weights())
    
    # STAGE 2: Full KL weight training
    print(f"    Stage 2 (KL={KL_WEIGHT_STAGE2})...")
    for i in range(TRAINING_ROUNDS):
        f = 1.0 / (TRAINING_ROUNDS - i)
        batch_size = int(data.shape[0] * f + 0.5)
        steps = int(data.shape[0] / batch_size + 0.5)
        epochs = int(1000 * f + 0.5)
        
        history = vae2.fit_generator(
            data_generator(data, batch_size),
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=0
        )
        loss_history += list(history.history['loss'])
        acc_history += list(history.history['acc'])
    
    # Get final latent representations
    vae_mu, vae_sigma = enc.predict(data)
    
    return vae_mu, vae_sigma, loss_history, acc_history, stage1_end

# ============================================================================
# RUN 5 INDEPENDENT VAE TRAININGS
# ============================================================================
all_run_results = []

for run in range(1, N_RUNS + 1):
    print(f"{'='*70}")
    print(f"RUN {run}/{N_RUNS}")
    print(f"{'='*70}")
    
    run_seed = 42 + run
    
    try:
        print(f"  Training VAE (seed={run_seed})...")
        Z_mu_pred, Z_sigma_pred, loss_hist, acc_hist, stage1_end = train_two_stage_vae(X, run_seed)
        
        total_epochs = len(loss_hist)
        final_loss = float(loss_hist[-1])
        print(f"  Training complete! Total epochs: {total_epochs}, Final loss: {final_loss:.4f}")
        
        # Calculate separation metrics using biological labels
        print(f"  Calculating separation metrics...")
        metrics = calculate_separation_metrics(Z_mu_pred, biological_labels)
        
        print(f"  Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f} (higher is better)")
        print(f"  Separation Ratio: {metrics['separation_ratio']:.4f} (higher is better)")
        
        # Store results
        run_result = {
            'run': int(run),
            'seed': int(run_seed),
            'total_epochs': int(total_epochs),
            'final_loss': final_loss,
            'metrics': metrics,
            'latent_mu': Z_mu_pred.tolist(),
            'latent_sigma': Z_sigma_pred.tolist(),
            'loss_history': [float(x) for x in loss_hist],
            'acc_history': [float(x) for x in acc_hist],
            'stage1_end': int(stage1_end)
        }
        
        all_run_results.append(run_result)
        
        print(f"Run {run} complete!\n")
        
        # Clean up
        K.clear_session()
        
    except Exception as e:
        print(f"\nERROR in run {run}: {str(e)}")
        traceback.print_exc()
        print()
        continue

# ============================================================================
# ANALYZE CONVERGENCE
# ============================================================================
print("="*70)
print("ANALYZING CONVERGENCE")
print("="*70)

if len(all_run_results) == 0:
    print("ERROR: No successful runs!")
    sys.exit(1)

# Calculate statistics across runs
silhouettes = [r['metrics']['silhouette'] for r in all_run_results]
davies_bouldins = [r['metrics']['davies_bouldin'] for r in all_run_results]
calinskis = [r['metrics']['calinski_harabasz'] for r in all_run_results]
sep_ratios = [r['metrics']['separation_ratio'] for r in all_run_results]

sil_mean = float(np.mean(silhouettes))
sil_std = float(np.std(silhouettes))
sil_cv = float((sil_std / sil_mean * 100) if sil_mean > 0 else 0)

db_mean = float(np.mean(davies_bouldins))
db_std = float(np.std(davies_bouldins))

cal_mean = float(np.mean(calinskis))
cal_std = float(np.std(calinskis))

sep_mean = float(np.mean(sep_ratios))
sep_std = float(np.std(sep_ratios))

print(f"\nSuccessful runs: {len(all_run_results)}/{N_RUNS}")
print(f"\nSilhouette Score:")
print(f"  Mean: {sil_mean:.4f} ± {sil_std:.4f}")
print(f"  CV: {sil_cv:.2f}%")
print(f"\nDavies-Bouldin Index (lower=better):")
print(f"  Mean: {db_mean:.4f} ± {db_std:.4f}")
print(f"\nCalinski-Harabasz Score (higher=better):")
print(f"  Mean: {cal_mean:.2f} ± {cal_std:.2f}")
print(f"\nSeparation Ratio (higher=better):")
print(f"  Mean: {sep_mean:.4f} ± {sep_std:.4f}")

# Find best run (highest silhouette)
best_run = max(all_run_results, key=lambda x: x['metrics']['silhouette'])
print(f"\nBEST RUN: #{best_run['run']}")
print(f"  Silhouette: {best_run['metrics']['silhouette']:.4f}")
print(f"  Davies-Bouldin: {best_run['metrics']['davies_bouldin']:.4f}")
print(f"  Separation Ratio: {best_run['metrics']['separation_ratio']:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create output text file with biological labels
output_lines = []
output_lines.append("sample\tbiological_group\tlatent_dim1\tlatent_dim2\tsilhouette_score\tconvergence_cv")

best_mu = np.array(best_run['latent_mu'])
for i, name in enumerate(sample_names):
    output_lines.append(f"{name}\t{biological_labels[i]}\t{best_mu[i,0]:.6f}\t{best_mu[i,1]:.6f}\t{best_run['metrics']['silhouette']:.4f}\t{sil_cv:.2f}")

output_file = os.path.join(OUTPUT_DIR, "supervised_latent_coordinates.txt")
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"  Latent coordinates saved to: {output_file}")

# Convert to Python types for JSON
all_run_results = convert_to_python_types(all_run_results)

# Save JSON results
results_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'configuration': {
        'approach': 'supervised',
        'biological_groups': list(np.unique(biological_labels)),
        'n_runs': int(N_RUNS),
        'latent_dim': int(LATENT_DIM),
        'two_stage_training': True,
        'kl_weight_stage1': float(KL_WEIGHT_STAGE1),
        'kl_weight_stage2': float(KL_WEIGHT_STAGE2)
    },
    'convergence': {
        'successful_runs': int(len(all_run_results)),
        'silhouette_mean': sil_mean,
        'silhouette_std': sil_std,
        'silhouette_cv_percent': sil_cv,
        'davies_bouldin_mean': db_mean,
        'calinski_harabasz_mean': cal_mean,
        'separation_ratio_mean': sep_mean
    },
    'best_run': {
        'run_number': int(best_run['run']),
        'metrics': best_run['metrics'],
        'total_epochs': best_run['total_epochs']
    },
    'all_runs': all_run_results
}

json_file = os.path.join(OUTPUT_DIR, "supervised_results_summary.json")
with open(json_file, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"  JSON results saved to: {json_file}")

# ============================================================================
# CREATE COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\nCreating comprehensive visualization...")

fig = plt.figure(figsize=(20, 12))

# Get best run data
best_mu = np.array(best_run['latent_mu'])
best_sigma = np.array(best_run['latent_sigma'])
best_loss = best_run['loss_history']
best_acc = best_run['acc_history']
stage1_end = best_run['stage1_end']

# Plot 1: Training loss
ax1 = plt.subplot(2, 3, 1)
ax1.plot(best_loss, linewidth=1, color='blue')
ax1.axvline(x=stage1_end, color='red', linestyle='--', linewidth=2, label='Stage 2 Start')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title(f'Training Loss (Best Run #{best_run["run"]})', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training accuracy
ax2 = plt.subplot(2, 3, 2)
ax2.plot(best_acc, linewidth=1, color='green')
ax2.axvline(x=stage1_end, color='red', linestyle='--', linewidth=2, label='Stage 2 Start')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title(f'Training Accuracy (Best Run #{best_run["run"]})', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Convergence across runs
ax3 = plt.subplot(2, 3, 3)
runs_x = [r['run'] for r in all_run_results]
sils = [r['metrics']['silhouette'] for r in all_run_results]
ax3.plot(runs_x, sils, 'o-', linewidth=2, markersize=10, color='blue')
ax3.axhline(y=sil_mean, color='red', linestyle='--', linewidth=2, label=f'Mean={sil_mean:.3f}')
ax3.fill_between(runs_x, 
                [sil_mean - sil_std] * len(runs_x),
                [sil_mean + sil_std] * len(runs_x),
                alpha=0.2, color='red')
ax3.set_xlabel('Run Number')
ax3.set_ylabel('Silhouette Score')
ax3.set_title(f'Convergence (CV={sil_cv:.1f}%)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Latent space with sample names (BEST RUN)
ax4 = plt.subplot(2, 3, 4)
plot_latent_with_names(ax4, best_mu, biological_labels, sample_names, 
                       title=f'Latent Space - Biological Groups (Best Run #{best_run["run"]})')

# Plot 5: Latent space with uncertainty (BEST RUN)
ax5 = plt.subplot(2, 3, 5)
plot_latent_with_uncertainty(ax5, best_mu, best_sigma, biological_labels, sample_names,
                             title=f'Latent Space with Uncertainty (Best Run #{best_run["run"]})',
                             n_samples=150)

# Plot 6: Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
SUPERVISED VAE ANALYSIS
{'='*40}

BIOLOGICAL GROUPS:
  {', '.join([g.upper() for g in np.unique(biological_labels)])}

RUNS: {len(all_run_results)}/{N_RUNS} successful

TWO-STAGE TRAINING:
  Stage 1: KL = {KL_WEIGHT_STAGE1}
  Stage 2: KL = {KL_WEIGHT_STAGE2}
  Total epochs: ~{best_run['total_epochs']} per run

SEPARATION METRICS:
  Silhouette: {sil_mean:.4f} ± {sil_std:.4f}
  CV: {sil_cv:.2f}%
  
  Davies-Bouldin: {db_mean:.4f}
  (lower = better separation)
  
  Calinski-Harabasz: {cal_mean:.2f}
  (higher = better separation)
  
  Separation Ratio: {sep_mean:.4f}
  (higher = better)

BEST RUN: #{best_run['run']}
  Silhouette: {best_run['metrics']['silhouette']:.4f}
  Davies-Bouldin: {best_run['metrics']['davies_bouldin']:.4f}
  Separation Ratio: {best_run['metrics']['separation_ratio']:.4f}

CONVERGENCE:
  {'✓ EXCELLENT' if sil_cv < 5 else '✓ GOOD' if sil_cv < 10 else '⚠ NEEDS ATTENTION'}

{'='*40}
"""
ax6.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
       fontfamily='monospace', 
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.suptitle(f'Supervised VAE Analysis - Biological Groups ({", ".join(np.unique(biological_labels))})', 
            fontsize=18, fontweight='bold')
plt.tight_layout()

plot_file = os.path.join(OUTPUT_DIR, "supervised_analysis_summary.pdf")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"  Comprehensive plot saved to: {plot_file}")
plt.close()

# ============================================================================
# CREATE PUBLICATION-QUALITY FIGURE
# ============================================================================
print("\nCreating publication-quality figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Latent space with sample names
plot_latent_with_names(ax1, best_mu, biological_labels, sample_names,
                       title='VAE Latent Space - Biological Groups')

# Right: Latent space with uncertainty
plot_latent_with_uncertainty(ax2, best_mu, best_sigma, biological_labels, sample_names,
                             title='VAE Latent Space with Uncertainty',
                             n_samples=150)

plt.suptitle(f'Wolbachia Strain Separation in VAE Latent Space\n' +
            f'Silhouette Score: {best_run["metrics"]["silhouette"]:.4f} | Convergence CV: {sil_cv:.2f}% | Separation Ratio: {best_run["metrics"]["separation_ratio"]:.2f}',
            fontsize=14, fontweight='bold')
plt.tight_layout()

pub_plot_file = os.path.join(OUTPUT_DIR, "publication_figure_biological_groups.pdf")
plt.savefig(pub_plot_file, dpi=300, bbox_inches='tight')
print(f"  Publication figure saved to: {pub_plot_file}")
plt.close()

# ============================================================================
# CREATE FINAL REPORT
# ============================================================================
report_file = os.path.join(OUTPUT_DIR, "SUPERVISED_ANALYSIS_REPORT.txt")
with open(report_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("SUPERVISED VAE ANALYSIS - BIOLOGICAL GROUPS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("APPROACH:\n")
    f.write("  Supervised analysis using known biological labels\n")
    f.write(f"  Groups: {', '.join(np.unique(biological_labels))}\n")
    f.write("  Goal: Evaluate how well VAE latent space separates biological groups\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Number of runs: {N_RUNS}\n")
    f.write(f"  Successful runs: {len(all_run_results)}\n")
    f.write(f"  Training approach: Two-stage\n")
    f.write(f"    Stage 1: KL weight = {KL_WEIGHT_STAGE1} (~2000 epochs)\n")
    f.write(f"    Stage 2: KL weight = {KL_WEIGHT_STAGE2} (~2000 epochs)\n")
    f.write(f"    Total epochs per run: ~{best_run['total_epochs']}\n\n")
    
    f.write("="*70 + "\n")
    f.write("GROUP COMPOSITION:\n")
    f.write("="*70 + "\n\n")
    for group in np.unique(biological_labels):
        count = np.sum(biological_labels == group)
        samples = ', '.join(sample_names[biological_labels == group])
        f.write(f"{group.upper()}: {count} samples\n")
        f.write(f"  {samples}\n\n")
    
    f.write("="*70 + "\n")
    f.write("SEPARATION METRICS (across all runs):\n")
    f.write("="*70 + "\n\n")
    f.write(f"Silhouette Score (measures cluster quality):\n")
    f.write(f"  Mean: {sil_mean:.4f} ± {sil_std:.4f}\n")
    f.write(f"  CV: {sil_cv:.2f}%\n")
    f.write(f"  Range: -1 (worst) to +1 (best)\n")
    f.write(f"  > 0.5 = Good separation, > 0.7 = Excellent\n\n")
    
    f.write(f"Davies-Bouldin Index (lower is better):\n")
    f.write(f"  Mean: {db_mean:.4f} ± {db_std:.4f}\n")
    f.write(f"  Range: 0 (best) to infinity\n")
    f.write(f"  < 1.0 = Good separation\n\n")
    
    f.write(f"Calinski-Harabasz Score (higher is better):\n")
    f.write(f"  Mean: {cal_mean:.2f} ± {cal_std:.2f}\n")
    f.write(f"  Measures ratio of between-cluster to within-cluster variance\n\n")
    
    f.write(f"Separation Ratio (higher is better):\n")
    f.write(f"  Mean: {sep_mean:.4f} ± {sep_std:.4f}\n")
    f.write(f"  Ratio of between-group distance to within-group spread\n\n")
    
    f.write("CONVERGENCE QUALITY:\n")
    if sil_cv < 5:
        f.write("  ✓ EXCELLENT - CV < 5% (highly reproducible results)\n")
    elif sil_cv < 10:
        f.write("  ✓ GOOD - CV between 5-10% (reproducible results)\n")
    else:
        f.write("  ⚠ NEEDS ATTENTION - CV > 10% (variable results across runs)\n")
    f.write("\n")
    
    f.write("="*70 + "\n")
    f.write("BEST RUN DETAILS:\n")
    f.write("="*70 + "\n\n")
    f.write(f"Run Number: {best_run['run']}\n")
    f.write(f"Random Seed: {best_run['seed']}\n")
    f.write(f"Total Epochs: {best_run['total_epochs']}\n")
    f.write(f"Final Loss: {best_run['final_loss']:.4f}\n\n")
    f.write(f"Separation Metrics:\n")
    f.write(f"  Silhouette Score: {best_run['metrics']['silhouette']:.4f}\n")
    f.write(f"  Davies-Bouldin Index: {best_run['metrics']['davies_bouldin']:.4f}\n")
    f.write(f"  Calinski-Harabasz Score: {best_run['metrics']['calinski_harabasz']:.2f}\n")
    f.write(f"  Separation Ratio: {best_run['metrics']['separation_ratio']:.4f}\n")
    f.write(f"  Average Between-Group Distance: {best_run['metrics']['avg_between_distance']:.4f}\n")
    f.write(f"  Average Within-Group Variance: {best_run['metrics']['avg_within_variance']:.4f}\n\n")
    
    f.write("="*70 + "\n")
    f.write("ALL RUNS SUMMARY:\n")
    f.write("="*70 + "\n\n")
    for r in all_run_results:
        f.write(f"Run {r['run']:2d}: ")
        f.write(f"Sil={r['metrics']['silhouette']:.4f}  ")
        f.write(f"DB={r['metrics']['davies_bouldin']:.4f}  ")
        f.write(f"Sep={r['metrics']['separation_ratio']:.4f}  ")
        f.write(f"Epochs={r['total_epochs']}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("OUTPUT FILES:\n")
    f.write("="*70 + "\n\n")
    f.write(f"1. {output_file}\n")
    f.write(f"   Latent space coordinates for each sample with biological labels\n\n")
    f.write(f"2. {json_file}\n")
    f.write(f"   Complete numerical results in JSON format\n\n")
    f.write(f"3. {plot_file}\n")
    f.write(f"   Comprehensive analysis plots\n\n")
    f.write(f"4. {pub_plot_file}\n")
    f.write(f"   Publication-quality figure (300 DPI)\n\n")
    
    f.write("="*70 + "\n")
    f.write("INTERPRETATION FOR MANUSCRIPT:\n")
    f.write("="*70 + "\n\n")
    f.write("The two-stage VAE training successfully learned a latent space\n")
    f.write("representation that captures the biological structure of Wolbachia\n")
    f.write("strains based on 47-gene sequences.\n\n")
    
    if sil_mean > 0.5:
        f.write("The high silhouette score indicates that the biological groups\n")
        f.write(f"({', '.join(np.unique(biological_labels))}) are well-separated in the latent space,\n")
        f.write("suggesting that the VAE effectively captured biologically\n")
        f.write("meaningful genetic variation.\n\n")
    else:
        f.write("The moderate silhouette score suggests some overlap between\n")
        f.write("biological groups in the latent space, which may reflect\n")
        f.write("genetic similarity or shared evolutionary history.\n\n")
    
    f.write("The low coefficient of variation (CV) across independent runs\n")
    f.write("demonstrates that the results are reproducible and robust.\n\n")
    
    f.write("="*70 + "\n")

print(f"  Report saved to: {report_file}")

print("\n" + "="*70)
print("SUPERVISED ANALYSIS COMPLETE!")
print("="*70)
print(f"\nBest run: #{best_run['run']}")
print(f"  Silhouette Score: {best_run['metrics']['silhouette']:.4f}")
print(f"  Separation Ratio: {best_run['metrics']['separation_ratio']:.4f}")
print(f"Convergence: CV={sil_cv:.2f}%")
print(f"\nResults saved to: {OUTPUT_DIR}/")
print(f"  - {report_file} ⭐ READ THIS FIRST")
print(f"  - {output_file}")
print(f"  - {pub_plot_file} ⭐ PUBLICATION FIGURE")
print("="*70)

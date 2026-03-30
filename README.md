# Wolbachia Strain Differentiation using Variational Autoencoders

Custom VAE implementation for analyzing genetic structure in *Wolbachia* bacterial endosymbionts using 47-gene genomic alignments.

## Requirements

Python 3.8+; tensorflow 2.13.1; scikit-learn 1.3.2; numpy 1.24.3; matplotlib 3.7.1; scipy 1.10.1

## Usage

### 1. Convert PHYLIP alignment to one-hot encoding

```bash
python load_phylip.py <input.phy> <output.txt>
```

**Input:** PHYLIP-formatted alignment (sequential or interleaved)  
**Output:** One-hot encoded sequences (text file)

### 2. Run supervised VAE analysis

Evaluates how well the latent space separates predefined biological groups. Group labels (column 2 of the input file) are used to color the latent space and calculate separation metrics.

```bash
python supervised_vae_biological_groups.py <datafile.txt>
```

**Optional arguments:**
- `--runs N` : Number of independent runs (default: 5)
- `--latent-dim N` : Latent space dimensions (default: 2)

**Example:**
```bash
python supervised_vae_biological_groups.py wolbachia_47genes_strain_newIDs.txt --runs 5
```

**Output:** Results saved to `supervised_results/by_{category}/`
- Latent coordinates (TXT, JSON)
- Comprehensive analysis report (TXT)
- Publication-quality figures (PDF)
- Training metrics and convergence plots

### 3. Run unsupervised VAE analysis

Discovers latent space structure without using biological labels. Multiple clustering algorithms (K-Means, Agglomerative, GMM) are tested across a range of k values; the best solution is selected by silhouette score. Biological labels are used only in a post-hoc comparison table to aid interpretation.

```bash
python unsupervised_vae_analysis.py <datafile.txt>
```

**Optional arguments:**
- `--runs N` : Number of independent runs (default: 5)
- `--latent-dim N` : Latent space dimensions (default: 2)
- `--k-min N` : Minimum number of clusters to test (default: 2)
- `--k-max N` : Maximum number of clusters to test (default: 5)

**Example:**
```bash
python unsupervised_vae_analysis.py wolbachia_47genes_strain_newIDs.txt --runs 5 --k-min 2 --k-max 5
```

**Output:** Results saved to `unsupervised_results/`
- Latent coordinates with discovered cluster assignments (TXT, JSON)
- Comprehensive analysis report including post-hoc biological label comparison (TXT)
- Publication-quality figures (PDF)
- Silhouette score heatmap across all algorithm × k combinations (PDF)
- Training metrics and convergence plots

## Input Data Format

Both VAE scripts expect one-hot encoded data in the following format:

```
sample_name group [1,0,0,0] [0,1,0,0] [0,0,1,0] ...
```

Where:
- Column 1: Sample identifier
- Column 2: Biological group label (e.g., CER, CIN, SINGLE) — used by the supervised script; stored but not used during analysis by the unsupervised script
- Remaining columns: One-hot encoded nucleotide vectors

## Data Availability

Input genomic alignments and latent space coordinates are deposited in [ZENODO](https://zenodo.org/) under accession [https://doi.org/10.5281/zenodo.18431813](https://doi.org/10.5281/zenodo.18431813)

## Citation

If you use this code, please cite:

Corretto, E., Ragionieri, L., Wolfe, T.M., Palmieri, L., Serbina, L.S., Bruzzese, D.J., Klasson, L., Feder, J.L., Stauffer, C., and Schuler, H. (2026). Interspecific Horizontal Wolbachia Transmission Between Native and Invasive Fruit Flies: Tracing a Rapid Wolbachia Spread in Slow Motion. Current Biology[under review].

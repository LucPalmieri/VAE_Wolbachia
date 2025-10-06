# Wolbachia Species Delimitation VAE

This repository packages the Variational Auto-Encoder (VAE) workflow that was
shared as a standalone script so that the analysis can be easily reproduced and
version-controlled.  The code targets TensorFlow 2.x via the
`tensorflow.compat.v1` API to remain compatible with the original implementation
and makes it straightforward to run multiple model initialisations, collect
metrics, and archive the resulting figures.

## Repository layout

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── src/
│   └── vae_analysis.py
└── data/
    └── wolbachia_47genes_host_newIDS.txt
    └── wolbachia_47genes_unsupervised_newIDS.txt
    └── wolbachia_47genes_country_newIDS.txt
```

Generated plots are written to the `vae_results/` directory (ignored by Git).

## Dataset format

The script expects plain-text input files where each line encodes a sample
using the following structure:

```
<sample_id> <group_label> [v1,v2,...,vk] [v1,v2,...,vk] ...
```

The Wolbachia file `wolbachia_47genes_host_newIDs.txt`, for instance, contains
47 genes (columns) encoded as one-hot vectors.  Place the dataset in the `data/`
folder so the provided commands work as-is.

## Getting started

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-user>/Species_delimitation_Wolbachia.git
   cd Species_delimitation_Wolbachia
   ```

2. **Create an isolated Python environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the analysis**
   ```bash
   python -m src.vae_analysis \
       --data data/wolbachia_47genes_host_newIDs.txt \
       --runs 3 \
       --output-dir vae_results
   ```

   Use `--show-plot` to display intermediate figures during training and
   `--verbose` for more detailed logging.

5. **Review outputs**
   - `vae_results/` contains a timestamped PDF summary and per-run reports.
   - Console logs include per-run silhouette scores and aggregate statistics.

## Reproducibility tips

- The script seeds NumPy and TensorFlow (default seed: 42).  Override via the
  `--seed` flag to explore variability between runs.
- All hyperparameters (latent size, hidden layers, dropout, learning rate,
  training schedule) are defined inside `mk_model` within
  `src/vae_analysis.py`.  Adjust them there if needed.

## Contributing

Issues and pull requests are welcome.  Please open an issue before submitting
major changes so we can discuss the approach.  When contributing, remember to
run the analysis locally to ensure everything works end-to-end.

## License

This project is distributed under the terms of the MIT License.  See the
[LICENSE](LICENSE) file for details.

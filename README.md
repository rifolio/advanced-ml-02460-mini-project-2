# Mini-project 2 — ensemble VAE (`ensemble_vae.py`)

## Setup

Use Python 3.12 (see `.python-version`). From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision tqdm matplotlib numpy
```

The first run downloads MNIST into `data/` (ignored by git).

## CLI overview

The first argument is **mode** (required):

| Mode | Purpose |
|------|---------|
| `train` | Fit the VAE; writes `model.pt` and `run_meta.json` under `--experiment-folder`. |
| `eval` | Mean test ELBO (scalar sanity check). |
| `sample` | Save prior samples + one batch of reconstructions as PNGs under `--image-output-dir`. |
| `geodesics` | Latent scatter + geodesics: pull-back if $D=1$, **model-average** energy if $D>1$; PNG under `--image-output-dir`. |
| `cov` | Part B: mean CoV of Euclidean vs geodesic distances vs $D$ across retrained models; saves `partb_cov.png` (or `--cov-figure-name`). |

List all options:

```bash
python ensemble_vae.py --help
python ensemble_vae.py train --help
```

## Training

Example (Apple Silicon GPU):

```bash
python ensemble_vae.py train --device mps --epochs-per-decoder 150 --experiment-folder experiment_long
```

Example (CPU):

```bash
python ensemble_vae.py train --device cpu --epochs-per-decoder 50 --experiment-folder experiment
```

Training writes:

- `<experiment-folder>/model.pt` — weights  
- `<experiment-folder>/run_meta.json` — epochs and naming metadata used for figure filenames  

## Evaluation and quick checks (“testing”)

After training, use the **same** `--experiment-folder` as for `train`.

**Test-set ELBO** (higher is better; printed to the terminal):

```bash
python ensemble_vae.py eval --device mps --experiment-folder experiment_long
```

**Samples and reconstructions** (PNG files):

```bash
python ensemble_vae.py sample --device mps --experiment-folder experiment_long --image-output-dir report_images
```

Outputs look like `report_images/<prefix>_e<epochs>_samples.png` and `..._reconstruction.png`, where `<prefix>` is the experiment folder name (unless you pass `--figure-prefix`).

## Geodesics (Part A figures)

```bash
python ensemble_vae.py geodesics --device mps --experiment-folder experiment_long --image-output-dir report_images
```

Tune optimization if needed: `--geodesic-lr`, `--geodesic-steps`, `--num-pairs` (≥ 25 for the assignment), `--num-t` (points along each polyline). For ensemble decoders ($D>1$), use `--mc-samples` for the Monte Carlo energy (default 16).

## Typical full workflow

```bash
# 1) Train
python ensemble_vae.py train --device mps --epochs-per-decoder 150 --experiment-folder experiment_long

# 2) Check ELBO
python ensemble_vae.py eval --device mps --experiment-folder experiment_long

# 3) Figures
python ensemble_vae.py sample --device mps --experiment-folder experiment_long --image-output-dir report_images
python ensemble_vae.py geodesics --device mps --experiment-folder experiment_long --image-output-dir report_images
```

Use `--device cuda` on NVIDIA GPUs.

## Part B: sweep script (ensemble decoders × retrainings)

Train $D \in \{1,2,3\}$ decoders per VAE and $M$ independent runs with consistent folders and figure names:

```bash
chmod +x scripts/run_partb_experiments.sh   # once
./scripts/run_partb_experiments.sh train
./scripts/run_partb_experiments.sh sample
./scripts/run_partb_experiments.sh geodesics
```

Models go under `experiments/partb/d<D>_r<MM>/` (e.g. `d3_r07` = $D=3$, rerun 7). Each `run_meta.json` stores `num_decoders`, `rerun_index`, `num_reruns`, and `training_seed`. Figures use the folder basename as prefix, e.g. `report_images/partb/d3_r07_e150_geodesics.png`.

Shared latent pairs for all runs: set `GEODESIC_PAIRS_DIR` (default `experiments/partb/_shared_pairs`) so every model uses the same `geodesic_pairs.pt`.

Override examples:

```bash
DEVICE=mps EPOCHS_PER_DECODER=50 NUM_RERUNS=3 ./scripts/run_partb_experiments.sh train
EXPERIMENTS_ROOT=runs/my_sweep IMAGE_OUTPUT_ROOT=report_images/my_sweep ./scripts/run_partb_experiments.sh sample
```

Resume training without overwriting existing checkpoints (default `SKIP_IF_EXISTS=1`):

```bash
SKIP_IF_EXISTS=1 DEVICE=mps DECODER_SWEEP="2 3" ./scripts/run_partb_experiments.sh train
```
For windows:
```bash
PYTHON=python SKIP_IF_EXISTS=1 DEVICE=cpu DECODER_SWEEP="2 3" ./scripts/run_partb_experiments.sh train
```

**CoV plot** (after all `experiments/partb/d<D>_r<MM>/model.pt` exist; uses shared pairs under `experiments/partb/_shared_pairs` by default):

```bash
python ensemble_vae.py cov --device mps \
  --experiments-root experiments/partb \
  --decoder-sweep 1 2 3 --num-reruns 10 \
  --geodesic-pairs-dir experiments/partb/_shared_pairs \
  --image-output-dir report_images/partb
```

## Notes

- **Epoch count** in figure names comes from `run_meta.json` produced during training. If you copy in a `model.pt` without that file, pass a matching `--epochs-per-decoder` when running `sample` / `geodesics` so the `_e<number>_` suffix is correct.
- There is no separate automated test suite; use `eval` plus visual inspection of `sample` / `geodesics` outputs.

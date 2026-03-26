# Mini-project 2 — ensemble VAE (`ensemble_vae.py`)

## Setup

Use Python 3.12 (see `.python-version`). From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision tqdm matplotlib
```

The first run downloads MNIST into `data/` (ignored by git).

## CLI overview

The first argument is **mode** (required):

| Mode | Purpose |
|------|---------|
| `train` | Fit the VAE; writes `model.pt` and `run_meta.json` under `--experiment-folder`. |
| `eval` | Mean test ELBO (scalar sanity check). |
| `sample` | Save prior samples + one batch of reconstructions as PNGs under `--image-output-dir`. |
| `geodesics` | Latent scatter + pull-back geodesics (Part A); saves a PNG under `--image-output-dir`. |

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

Tune optimization if needed: `--geodesic-lr`, `--geodesic-steps`, `--num-pairs` (≥ 25 for the assignment), `--num-t` (points along each polyline).

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

## Notes

- **Epoch count** in figure names comes from `run_meta.json` produced during training. If you copy in a `model.pt` without that file, pass a matching `--epochs-per-decoder` when running `sample` / `geodesics` so the `_e<number>_` suffix is correct.
- There is no separate automated test suite; use `eval` plus visual inspection of `sample` / `geodesics` outputs.

# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

RUN_META_FILENAME = "run_meta.json"


def experiment_slug(experiment_folder):
    return os.path.basename(os.path.normpath(experiment_folder))


def load_run_meta(experiment_folder):
    path = os.path.join(experiment_folder, RUN_META_FILENAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_run_meta(experiment_folder, meta):
    os.makedirs(experiment_folder, exist_ok=True)
    path = os.path.join(experiment_folder, RUN_META_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def figure_filename(prefix, epochs, kind):
    """kind: samples | reconstruction | geodesics"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(prefix))
    return f"{safe}_e{int(epochs)}_{kind}.png"


def migrate_state_dict_to_ensemble(state_dict):
    """Map legacy single-decoder keys (decoder.*) to decoders.0.* for EnsembleVAE."""
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("decoders.") for k in state_dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith("decoder."):
            out["decoders.0." + k[len("decoder.") :]] = v
        else:
            out[k] = v
    return out

def geodesic_length(model, z_path):
    """True arc length — sum of decoded segment lengths after optimization."""
    with torch.no_grad():
        f = decoder_mean_flat(model, z_path)              # (N, 784)
        segment_lengths = torch.norm(f[1:] - f[:-1], dim=1)  # (N-1,)
    return segment_lengths.sum().item()


def euclidean_length(z_path):
    """Straight-line distance between endpoints in latent space."""
    return torch.norm(z_path[-1] - z_path[0]).item()


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


class EnsembleVAE(nn.Module):
    """
    VAE with multiple decoder networks; each training step uses one decoder
    drawn uniformly (ensemble training for Part B).
    """

    def __init__(self, prior, decoders, encoder):
        super().__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)
        self.encoder = encoder

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        idx = torch.randint(0, len(self.decoders), (1,)).item()
        dec = self.decoders[idx]
        return torch.mean(
            dec(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoders[0](z).sample()

    def forward(self, x):
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


def decoder_mean_flat(model, z, decoder_index=0):
    """
    Decoder mean f(z) as a flat vector for pull-back distances in image space.

    z: (batch, M) or (M,)
    decoder_index: which ensemble member (Part A uses 0).
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if hasattr(model, "decoders"):
        means = model.decoders[decoder_index].decoder_net(z)
    else:
        means = model.decoder.decoder_net(z)
    return means.view(z.shape[0], -1)


def pullback_curve_energy(model, z_path):
    """
    Discrete pull-back energy sum_i ||f(z_{i+1}) - f(z_i)||^2 for the decoder mean f.

    z_path: (K+1, M) with requires_grad as needed for interior points.
    """
    f = decoder_mean_flat(model, z_path)
    return torch.sum((f[1:] - f[:-1]).pow(2))


def ensemble_curve_energy(model, z_path, mc_samples=16):
    """
    Model-average discrete curve energy (Part B, Eq. 1):
    sum_i E_{l,k}[ ||f_l(z_i) - f_k(z_{i+1})||^2 ], approximated by Monte Carlo.

    When D=1 this equals pullback_curve_energy (same f_0 at both endpoints of each segment).
    """
    if not hasattr(model, "decoders"):
        return pullback_curve_energy(model, z_path)
    d_dec = len(model.decoders)
    if d_dec == 1:
        return pullback_curve_energy(model, z_path)

    k_seg = z_path.shape[0] - 1
    if k_seg <= 0:
        return z_path.sum() * 0.0

    device = z_path.device
    total = z_path.new_zeros(())
    for i in range(k_seg):
        zi = z_path[i : i + 1]
        zj = z_path[i + 1 : i + 2]
        seg_sum = z_path.new_zeros(())
        for _ in range(mc_samples):
            li = int(torch.randint(0, d_dec, (1,), device=device).item())
            ki = int(torch.randint(0, d_dec, (1,), device=device).item())
            fi = decoder_mean_flat(model, zi, li)
            fj = decoder_mean_flat(model, zj, ki)
            seg_sum = seg_sum + (fi - fj).pow(2).sum()
        total = total + seg_sum / float(mc_samples)
    return total


def decoder_uncertainty_grid(model, xlim, ylim, resolution=80, device="cpu"):
    """
    Compute per-grid-point decoder std dev (mean over pixels) for the ensemble heatmap.
    Returns (std_map, xs, ys) where std_map is (resolution, resolution).
    Returns (None, None, None) if the model has fewer than 2 decoders.
    """
    if not hasattr(model, "decoders") or len(model.decoders) < 2:
        return None, None, None
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    z_flat = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        outputs = []
        for dec in model.decoders:
            f = dec.decoder_net(z_flat).view(z_flat.shape[0], -1)  # (N, 784)
            outputs.append(f.cpu().numpy())
    outputs = np.stack(outputs, axis=0)          # (D, N, 784)
    std_map = outputs.std(axis=0).mean(axis=1).reshape(xx.shape)  # (H, W)
    return std_map, xs, ys


def optimize_geodesic(  
    model,
    z0,
    z1,
    num_interior,
    device,
    lr=0.05,
    steps=500,
):
    """
    Minimize pull-back energy over interior waypoints; endpoints fixed.

    z0, z1: (M,) on device
    Returns: (num_interior + 2, M) latent polyline
    """
    z0 = z0.detach().clone()
    z1 = z1.detach().clone()
    if num_interior <= 0:
        return torch.stack([z0, z1], dim=0)

    ts = torch.linspace(0.0, 1.0, num_interior + 2, device=device)[1:-1]
    init = torch.stack([(1.0 - t) * z0 + t * z1 for t in ts], dim=0)
    z_in = nn.Parameter(init.clone())
    opt = torch.optim.Adam([z_in], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        z_path = torch.cat([z0.unsqueeze(0), z_in, z1.unsqueeze(0)], dim=0)
        e = pullback_curve_energy(model, z_path)
        e.backward()
        opt.step()

    with torch.no_grad():
        z_path = torch.cat([z0.unsqueeze(0), z_in.detach(), z1.unsqueeze(0)], dim=0)
    return z_path


def optimize_ensemble_geodesic(
    model,
    z0,
    z1,
    num_interior,
    device,
    lr=0.05,
    steps=500,
    mc_samples=16,
):
    """
    Minimize ensemble (model-average) curve energy over interior waypoints; endpoints fixed.
    """
    z0 = z0.detach().clone()
    z1 = z1.detach().clone()
    if num_interior <= 0:
        return torch.stack([z0, z1], dim=0)

    ts = torch.linspace(0.0, 1.0, num_interior + 2, device=device)[1:-1]
    init = torch.stack([(1.0 - t) * z0 + t * z1 for t in ts], dim=0)
    z_in = nn.Parameter(init.clone())
    opt = torch.optim.Adam([z_in], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        z_path = torch.cat([z0.unsqueeze(0), z_in, z1.unsqueeze(0)], dim=0)
        e = ensemble_curve_energy(model, z_path, mc_samples=mc_samples)
        e.backward()
        opt.step()

    with torch.no_grad():
        z_path = torch.cat([z0.unsqueeze(0), z_in.detach(), z1.unsqueeze(0)], dim=0)
    return z_path


def curve_energy_at_path(model, z_path, use_ensemble, mc_samples):
    """Energy of a fixed polyline (for reporting distance after optimization)."""
    if use_ensemble:
        return ensemble_curve_energy(model, z_path, mc_samples=mc_samples)
    return pullback_curve_energy(model, z_path)


def build_ensemble_vae(M, num_decoders, new_encoder_fn, new_decoder_fn, device):
    """num_decoders >= 1."""
    prior = GaussianPrior(M)
    encoder = GaussianEncoder(new_encoder_fn())
    decoders = [GaussianDecoder(new_decoder_fn()) for _ in range(num_decoders)]
    return EnsembleVAE(prior, decoders, encoder).to(device)


def load_model_weights(model, path, map_location):
    state = torch.load(path, map_location=map_location)
    state = migrate_state_dict_to_ensemble(state)
    model.load_state_dict(state, strict=True)


def resolved_training_seed(args):
    if getattr(args, "training_seed", None) is not None:
        return args.training_seed
    return args.seed + args.rerun_index * 100_003


def run_cov_mode(
    args,
    M,
    new_encoder,
    new_decoder,
    mnist_test_loader,
):
    """
    Part B: CoV of Euclidean vs geodesic distances across M retrained VAEs, per D.
    """
    device_t = torch.device(args.device)
    experiments_root = args.experiments_root
    decoder_sweep = args.decoder_sweep if args.decoder_sweep is not None else [1, 2, 3]
    num_reruns = args.num_reruns
    num_pairs = args.num_pairs
    pair_dir = args.geodesic_pairs_dir or os.path.join(
        experiments_root, "_shared_pairs"
    )
    os.makedirs(pair_dir, exist_ok=True)
    pair_path = os.path.join(pair_dir, "geodesic_pairs.pt")

    num_points = len(mnist_test_loader.dataset)
    try:
        pair_indices = torch.load(pair_path)
        if pair_indices.shape != (num_pairs, 2):
            raise ValueError("shape mismatch")
    except (FileNotFoundError, ValueError):
        torch.manual_seed(args.seed)
        pair_indices = torch.randint(0, num_points, (num_pairs, 2))
        for i in range(num_pairs):
            while pair_indices[i, 0] == pair_indices[i, 1]:
                pair_indices[i, 1] = torch.randint(0, num_points, (1,))
        torch.save(pair_indices, pair_path)

    num_interior = max(0, args.num_t - 2)
    mc = args.mc_samples

    cov_eucl_mean = []
    cov_geo_mean = []
    d_list = []

    for D in decoder_sweep:
        d_list.append(D)
        eucl_dists = []  # list over successful runs: each is (num_pairs,)
        geo_dists = []

        for r in range(num_reruns):
            exp_folder = os.path.join(experiments_root, f"d{D}_r{r:02d}")
            model_path = os.path.join(exp_folder, "model.pt")
            if not os.path.isfile(model_path):
                print(f"skip missing model: {model_path}")
                continue

            meta = load_run_meta(exp_folder)
            num_decoders = int(meta.get("num_decoders", D))
            if num_decoders != D:
                print(
                    f"warning: {exp_folder} meta num_decoders={num_decoders} != D={D}"
                )

            model = build_ensemble_vae(
                M, num_decoders, new_encoder, new_decoder, device_t
            )
            load_model_weights(model, model_path, device_t)
            model.eval()

            zs = []
            with torch.no_grad():
                for x, _y in mnist_test_loader:
                    x = x.to(device_t)
                    zs.append(model.encoder(x).mean.cpu())
            z_all = torch.cat(zs, dim=0)

            e_row = []
            g_row = []
            for p in range(num_pairs):
                ia = int(pair_indices[p, 0])
                ib = int(pair_indices[p, 1])
                za = z_all[ia].to(device_t)
                zb = z_all[ib].to(device_t)
                eucl = torch.norm(za - zb).item()
                e_row.append(eucl)

                if num_decoders > 1:
                    path = optimize_ensemble_geodesic(
                        model,
                        za,
                        zb,
                        num_interior,
                        device_t,
                        lr=args.geodesic_lr,
                        steps=args.geodesic_steps,
                        mc_samples=mc,
                    )
                else:
                    path = optimize_geodesic(
                        model,
                        za,
                        zb,
                        num_interior,
                        device_t,
                        lr=args.geodesic_lr,
                        steps=args.geodesic_steps,
                    )
                with torch.no_grad():
                    E = curve_energy_at_path(
                        model,
                        path,
                        use_ensemble=(num_decoders > 1),
                        mc_samples=mc,
                    )
                g_row.append(torch.sqrt(E + 1e-12).item())

            eucl_dists.append(np.array(e_row, dtype=np.float64))
            geo_dists.append(np.array(g_row, dtype=np.float64))

        if len(eucl_dists) == 0:
            print(f"warning: no models for D={D}, skipping CoV")
            cov_eucl_mean.append(float("nan"))
            cov_geo_mean.append(float("nan"))
            continue

        stack_e = np.stack(eucl_dists, axis=0)
        stack_g = np.stack(geo_dists, axis=0)
        cov_e = []
        cov_g = []
        for p in range(num_pairs):
            cov_e.append(np.std(stack_e[:, p]) / (np.mean(stack_e[:, p]) + 1e-12))
            cov_g.append(np.std(stack_g[:, p]) / (np.mean(stack_g[:, p]) + 1e-12))
        cov_eucl_mean.append(float(np.mean(cov_e)))
        cov_geo_mean.append(float(np.mean(cov_g)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        d_list,
        cov_eucl_mean,
        marker="o",
        label="Euclidean distance CoV",
    )
    ax.plot(
        d_list,
        cov_geo_mean,
        marker="s",
        label="Geodesic distance CoV",
    )
    ax.set_xlabel("Number of ensemble decoders $D$")
    ax.set_ylabel("Mean CoV over pairs")
    ax.set_title("Distance reliability vs ensemble size (across VAE retrainings)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(args.image_output_dir, exist_ok=True)
    out_path = os.path.join(args.image_output_dir, args.cov_figure_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved CoV plot to {out_path}")
    print("Per-D mean CoV (Euclidean):", list(zip(d_list, cov_eucl_mean)))
    print("Per-D mean CoV (geodesic):", list(zip(d_list, cov_geo_mean)))


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--image-output-dir",
        type=str,
        default="report_images",
        help="directory for all PNG outputs (samples, reconstructions, geodesics) (default: %(default)s)",
    )
    parser.add_argument(
        "--figure-prefix",
        type=str,
        default=None,
        help="filename prefix for figures; default is the experiment folder name (e.g. experiment_long)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=1,
        metavar="D",
        help="ensemble size D: number of decoder networks (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="total points along each geodesic polyline including endpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=25,
        metavar="N",
        help="number of random latent pairs for geodesics (Part A: at least 25) (default: %(default)s)",
    )
    parser.add_argument(
        "--geodesic-lr",
        type=float,
        default=0.05,
        help="learning rate for geodesic waypoint optimization (default: %(default)s)",
    )
    parser.add_argument(
        "--geodesic-steps",
        type=int,
        default=500,
        metavar="N",
        help="optimizer steps per geodesic (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base RNG seed; geodesic pair sampling uses this (default: %(default)s)",
    )
    parser.add_argument(
        "--rerun-index",
        type=int,
        default=0,
        metavar="N",
        help="index of this VAE retraining (0..M-1); stored in run_meta for Part B (default: %(default)s)",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        metavar="N",
        help="torch RNG seed for training; default: --seed + --rerun-index * 100003",
    )
    parser.add_argument(
        "--geodesic-pairs-dir",
        type=str,
        default=None,
        help="where to read/write geodesic_pairs.pt (default: --experiment-folder)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=16,
        metavar="S",
        help="Monte Carlo samples for ensemble curve energy (Part B) (default: %(default)s)",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="experiments/partb",
        help="cov mode: directory containing d<D>_r<RR> experiment folders (default: %(default)s)",
    )
    parser.add_argument(
        "--decoder-sweep",
        nargs="*",
        type=int,
        default=None,
        help="cov mode: list of D values (default: 1 2 3)",
    )
    parser.add_argument(
        "--cov-figure-name",
        type=str,
        default="partb_cov.png",
        help="cov mode: output PNG filename under --image-output-dir (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":
        train_seed = resolved_training_seed(args)
        torch.manual_seed(train_seed)

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = build_ensemble_vae(
            M, args.num_decoders, new_encoder, new_decoder, device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )
        run_meta = {
            "epochs_per_decoder": args.epochs_per_decoder,
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "num_decoders": args.num_decoders,
            "rerun_index": args.rerun_index,
            "num_reruns": args.num_reruns,
            "training_seed": train_seed,
            "experiment_folder": experiments_folder,
            "figure_prefix": args.figure_prefix or experiment_slug(experiments_folder),
            "trained_at_unix": time.time(),
        }
        save_run_meta(experiments_folder, run_meta)

    elif args.mode == "sample":
        device_t = torch.device(device)
        meta = load_run_meta(args.experiment_folder)
        epochs = int(meta.get("epochs_per_decoder", args.epochs_per_decoder))
        num_decoders = int(meta.get("num_decoders", args.num_decoders))
        prefix = (
            args.figure_prefix
            or meta.get("figure_prefix")
            or experiment_slug(args.experiment_folder)
        )
        os.makedirs(args.image_output_dir, exist_ok=True)
        path_samples = os.path.join(
            args.image_output_dir, figure_filename(prefix, epochs, "samples")
        )
        path_recon = os.path.join(
            args.image_output_dir, figure_filename(prefix, epochs, "reconstruction")
        )

        model = build_ensemble_vae(
            M, num_decoders, new_encoder, new_decoder, device_t
        )
        load_model_weights(
            model,
            os.path.join(args.experiment_folder, "model.pt"),
            device_t,
        )
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), path_samples)

            data = next(iter(mnist_test_loader))[0].to(device_t)
            recon = model.decoders[0](model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), path_recon
            )
        print(f"Saved samples to {path_samples}")
        print(f"Saved reconstructions to {path_recon}")

    elif args.mode == "eval":
        device_t = torch.device(device)
        meta = load_run_meta(args.experiment_folder)
        if meta:
            print(
                f"Run meta: epochs_per_decoder={meta.get('epochs_per_decoder')}, "
                f"figure_prefix={meta.get('figure_prefix')}, "
                f"num_decoders={meta.get('num_decoders')}, "
                f"rerun_index={meta.get('rerun_index')}"
            )
        num_decoders = int(meta.get("num_decoders", args.num_decoders))
        model = build_ensemble_vae(
            M, num_decoders, new_encoder, new_decoder, device_t
        )
        load_model_weights(
            model,
            os.path.join(args.experiment_folder, "model.pt"),
            device_t,
        )
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device_t)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        torch.manual_seed(args.seed)
        device_t = torch.device(device)
        meta = load_run_meta(args.experiment_folder)
        epochs = int(meta.get("epochs_per_decoder", args.epochs_per_decoder))
        prefix = (
            args.figure_prefix
            or meta.get("figure_prefix")
            or experiment_slug(args.experiment_folder)
        )
        os.makedirs(args.image_output_dir, exist_ok=True)
        path_geodesics = os.path.join(
            args.image_output_dir, figure_filename(prefix, epochs, "geodesics")
        )

        num_decoders = int(meta.get("num_decoders", args.num_decoders))
        model = build_ensemble_vae(
            M, num_decoders, new_encoder, new_decoder, device_t
        )
        state_path = os.path.join(args.experiment_folder, "model.pt")
        load_model_weights(model, state_path, device_t)
        model.eval()

        num_interior = max(0, args.num_t - 2)

        zs, ys = [], []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device_t)
                q = model.encoder(x)
                zs.append(q.mean.cpu())
                ys.append(y)
        z_train = torch.cat(zs, dim=0)
        y_train = torch.cat(ys, dim=0)

        num_pairs = args.num_pairs
        num_points = z_train.shape[0]

        pair_dir = args.geodesic_pairs_dir or args.experiment_folder
        os.makedirs(pair_dir, exist_ok=True)
        pair_path = os.path.join(pair_dir, "geodesic_pairs.pt")
        try:
            pair_indices = torch.load(pair_path)
            if pair_indices.shape != (num_pairs, 2):
                print(f"Warning: Loaded pair indices shape {pair_indices.shape} does not match expected {(num_pairs, 2)}. Resampling.")
                raise ValueError("Invalid pair indices shape")
        except (FileNotFoundError, ValueError):
            pair_indices = torch.randint(0, num_points, (num_pairs, 2))
            for i in range(num_pairs):
                while pair_indices[i, 0] == pair_indices[i, 1]:
                    pair_indices[i, 1] = torch.randint(0, num_points, (1,))
            torch.save(pair_indices, pair_path)

        geodesics_xy = []
        z_endpoints = []  # Store (z0, z1) pairs for plotting straight lines
        geo_lengths  = []
        eucl_lengths = []
        use_ensemble_geo = num_decoders > 1
        for i in tqdm(range(num_pairs), desc="geodesics"):
            z0 = z_train[pair_indices[i, 0]].to(device_t)
            z1 = z_train[pair_indices[i, 1]].to(device_t)
            z_endpoints.append((z0.cpu().numpy(), z1.cpu().numpy()))
            if use_ensemble_geo:
                path = optimize_ensemble_geodesic(
                    model,
                    z0,
                    z1,
                    num_interior,
                    device_t,
                    lr=args.geodesic_lr,
                    steps=args.geodesic_steps,
                    mc_samples=args.mc_samples,
                )
            else:
                path = optimize_geodesic(
                    model,
                    z0,
                    z1,
                    num_interior,
                    device_t,
                    lr=args.geodesic_lr,
                    steps=args.geodesic_steps,
                )
            geodesics_xy.append(path.cpu().numpy())
            geo_lengths.append(geodesic_length(model, path))
            eucl_lengths.append(euclidean_length(path))
        
        print(f"\n{'Pair':<6} {'Euclidean':>12} {'Geodesic':>12}")
        print("-" * 52)
        for i in range(num_pairs):
            print(f"{i:<6} {eucl_lengths[i]:>12.4f} {geo_lengths[i]:>12.4f}")
        
        z_train_np = z_train.numpy()
        y_train_np = y_train.numpy()

        pad = 0.5
        xlim = (z_train_np[:, 0].min() - pad, z_train_np[:, 0].max() + pad)
        ylim = (z_train_np[:, 1].min() - pad, z_train_np[:, 1].max() + pad)

        fig, ax = plt.subplots(figsize=(7, 6))

        # Uncertainty heatmap background for ensemble plots
        if use_ensemble_geo:
            std_map, _xs, _ys = decoder_uncertainty_grid(
                model, xlim, ylim, resolution=80, device=device_t
            )
            if std_map is not None:
                im = ax.imshow(
                    std_map,
                    origin="lower",
                    extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                    aspect="auto",
                    cmap="plasma",
                    alpha=0.85,
                )
                cbar = fig.colorbar(im, ax=ax, pad=0.02)
                cbar.set_label("Decoder Uncertainty (Std Dev)", fontsize=9)

        # Scatter latent data points
        class_colors = ["#5599ff", "#ff8833", "#44bb66"]
        for c in range(num_classes):
            mask = y_train_np == c
            ax.scatter(
                z_train_np[mask, 0], z_train_np[mask, 1],
                s=5, alpha=0.4, color=class_colors[c], label=f"Class {c}",
            )

        # Geodesic curves, Euclidean reference lines, and endpoint markers
        curve_colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_pairs)))
        ep_line_color = "white" if use_ensemble_geo else "gray"
        ep_marker_color = "white" if use_ensemble_geo else "black"
        ep_xs, ep_ys = [], []
        for idx, (path, (z0, z1)) in enumerate(zip(geodesics_xy, z_endpoints)):
            c = curve_colors[idx % len(curve_colors)]
            ax.plot(
                [z0[0], z1[0]], [z0[1], z1[1]],
                color=ep_line_color, linestyle="--", lw=0.8, alpha=0.5,
                label="Euclidean Path" if idx == 0 else None,
            )
            ax.plot(
                path[:, 0], path[:, 1], color=c, lw=1.5, alpha=0.9,
                label="Geodesic Path" if idx == 0 else None,
            )
            ep_xs += [z0[0], z1[0]]
            ep_ys += [z0[1], z1[1]]

        ax.scatter(
            ep_xs, ep_ys, s=40, facecolors="none",
            edgecolors=ep_marker_color, linewidths=1.2, zorder=5,
            label="Start/End Points",
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("z1", fontsize=11)
        ax.set_ylabel("z2", fontsize=11)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.7)
        if use_ensemble_geo:
            ax.set_title("Geodesic For Ensemble VAE", fontsize=12)
        else:
            ax.set_title("Geodesic in Latent Space", fontsize=12)
        fig.tight_layout()
        fig.savefig(path_geodesics, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {path_geodesics}")

    elif args.mode == "cov":
        run_cov_mode(args, M, new_encoder, new_decoder, mnist_test_loader)

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
                model = model
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


def decoder_mean_flat(model, z):
    """
    Decoder mean f(z) as a flat vector for pull-back distances in image space.

    z: (batch, M) or (M,)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    means = model.decoder.decoder_net(z)
    return means.view(z.shape[0], -1)


def pullback_curve_energy(model, z_path):
    """
    Discrete pull-back energy sum_i ||f(z_{i+1}) - f(z_i)||^2 for the decoder mean f.

    z_path: (K+1, M) with requires_grad as needed for interior points.
    """
    f = decoder_mean_flat(model, z_path)
    return torch.sum((f[1:] - f[:-1]).pow(2))


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
        choices=["train", "sample", "eval", "geodesics"],
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
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
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
        help="RNG seed for latent pair sampling (default: %(default)s)",
    )
    parser.add_argument(
        "--std-grid-size",
        type=int,
        default=120,
        metavar="N",
        help="resolution of latent grid for std background (default: %(default)s)",
    )
    parser.add_argument(
        "--std-pad",
        type=float,
        default=0.5,
        help="padding around latent cloud for std background grid (default: %(default)s)",
    )
    parser.add_argument(
        "--std-mc-samples",
        type=int,
        default=16,
        metavar="N",
        help="number of latent perturbation samples per grid point for std map (default: %(default)s)",
    )
    parser.add_argument(
        "--std-latent-noise",
        type=float,
        default=0.25,
        help="std of latent perturbations used for std map (default: %(default)s)",
    )
    parser.add_argument(
        "--std-batch-size",
        type=int,
        default=1024,
        metavar="N",
        help="number of grid points per forward batch for std map (default: %(default)s)",
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
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
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
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )
        run_meta = {
            "epochs_per_decoder": args.epochs_per_decoder,
            "latent_dim": args.latent_dim,
            "batch_size": args.batch_size,
            "experiment_folder": experiments_folder,
            "figure_prefix": args.figure_prefix or experiment_slug(experiments_folder),
            "trained_at_unix": time.time(),
        }
        save_run_meta(experiments_folder, run_meta)

    elif args.mode == "sample":
        device_t = torch.device(device)
        meta = load_run_meta(args.experiment_folder)
        epochs = int(meta.get("epochs_per_decoder", args.epochs_per_decoder))
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

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device_t)
        model.load_state_dict(
            torch.load(
                os.path.join(args.experiment_folder, "model.pt"),
                map_location=device_t,
            )
        )
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), path_samples)

            data = next(iter(mnist_test_loader))[0].to(device_t)
            recon = model.decoder(model.encoder(data).mean).mean
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
                f"figure_prefix={meta.get('figure_prefix')}"
            )
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device_t)
        model.load_state_dict(
            torch.load(
                os.path.join(args.experiment_folder, "model.pt"),
                map_location=device_t,
            )
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

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device_t)
        state_path = os.path.join(args.experiment_folder, "model.pt")
        model.load_state_dict(torch.load(state_path, map_location=device_t))
        model.eval()

        num_interior = max(0, args.num_t - 2)

        zs, ys = [], []
        with torch.no_grad():
            for x, y in mnist_train_loader:
                x = x.to(device_t)
                q = model.encoder(x)
                zs.append(q.mean.cpu())
                ys.append(y)
        z_train = torch.cat(zs, dim=0).numpy()
        y_train = torch.cat(ys, dim=0).numpy()

        num_pairs = args.num_pairs
        z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device_t)

        if z_train_t.shape[0] < 2:
            raise RuntimeError("Need at least 2 encoded points to form geodesic pairs.")

        if z_train_t.shape[0] < 2 * num_pairs:
            print(
                f"Warning: requested {num_pairs} disjoint pairs, but only "
                f"{z_train_t.shape[0]} encoded points are available. Sampling pairs with replacement."
            )
            pair_idx = torch.randint(
                low=0,
                high=z_train_t.shape[0],
                size=(num_pairs, 2),
                device=device_t,
            )
        else:
            perm = torch.randperm(z_train_t.shape[0], device=device_t)
            pair_idx = perm[: 2 * num_pairs].view(num_pairs, 2)

        geodesics_xy = []
        for i in tqdm(range(num_pairs), desc="geodesics"):
            z0 = z_train_t[pair_idx[i, 0]]
            z1 = z_train_t[pair_idx[i, 1]]
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

        zmin = z_train.min(axis=0)
        zmax = z_train.max(axis=0)
        x_lo, y_lo = zmin[0] - args.std_pad, zmin[1] - args.std_pad
        x_hi, y_hi = zmax[0] + args.std_pad, zmax[1] + args.std_pad
        gx = torch.linspace(x_lo, x_hi, args.std_grid_size, device=device_t)
        gy = torch.linspace(y_lo, y_hi, args.std_grid_size, device=device_t)
        mx, my = torch.meshgrid(gx, gy, indexing="xy")
        grid_z = torch.stack([mx.reshape(-1), my.reshape(-1)], dim=1)
        with torch.no_grad():
            std_vals = []
            for start in range(0, grid_z.shape[0], args.std_batch_size):
                z_chunk = grid_z[start : start + args.std_batch_size]
                b = z_chunk.shape[0]
                eps = torch.randn(
                    b,
                    args.std_mc_samples,
                    M,
                    device=device_t,
                ) * args.std_latent_noise
                z_mc = (z_chunk[:, None, :] + eps).reshape(-1, M)
                means_mc = model.decoder.decoder_net(z_mc).view(b, args.std_mc_samples, -1)
                std_chunk = means_mc.std(dim=1, unbiased=False).mean(dim=1)
                std_vals.append(std_chunk)
            std_map = torch.cat(std_vals, dim=0)
            std_map = std_map.view(args.std_grid_size, args.std_grid_size).cpu().numpy()
            vmin, vmax = np.percentile(std_map, [2.0, 98.0])

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            std_map.T,
            origin="lower",
            extent=[x_lo, x_hi, y_lo, y_hi],
            cmap="viridis",
            alpha=0.85,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Standard deviation of pixel values")
        for c in range(num_classes):
            mask = y_train == c
            ax.scatter(
                z_train[mask, 0],
                z_train[mask, 1],
                s=10,
                alpha=0.7,
                label=f"class {c}",
            )
        pair_colors = plt.cm.tab20(
            torch.linspace(0, 1, max(1, len(geodesics_xy))).cpu().numpy()
        )
        for i, path in enumerate(geodesics_xy):
            color = pair_colors[i]
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                lw=0.95,
                alpha=0.75,
                label="Pullback geodesic" if i == 0 else None,
            )
            ax.plot(
                [path[0, 0], path[-1, 0]],
                [path[0, 1], path[-1, 1]],
                linestyle=":",
                color=color,
                lw=1.1,
                alpha=0.9,
                label="Straight line" if i == 0 else None,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.legend(loc="best", fontsize=9)
        ax.set_title("Encoder means + pull-back geodesics (decoder mean)")
        fig.tight_layout()
        fig.savefig(path_geodesics, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {path_geodesics}")

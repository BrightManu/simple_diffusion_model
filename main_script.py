# main.py
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# dataset
def dino_dataset(n=8000):
    df = pd.read_csv("https://raw.githubusercontent.com/tanelp/tiny-diffusion/master/static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].to_numpy()
    y = df["y"].iloc[ix].to_numpy()

    x = x + rng.normal(size=n) * 0.15
    y = y + rng.normal(size=n) * 0.15

    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4

    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


# Positional embeddings
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)

# Define Model

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal"):
        super().__init__()
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        return self.joint_mlp(x)


# Noise Scheduler

class NoiseScheduler():
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "cosine":
            self.betas = self.cosine_beta_schedule(num_timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()

        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * self.alphas_cumprod_prev.sqrt() / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * self.alphas.sqrt() / (1. - self.alphas_cumprod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(torch.tensor(betas, dtype=torch.float32), 1e-5, 0.999)

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        return s1 * x_start + s2 * x_noise

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t].reshape(-1, 1)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t].reshape(-1, 1)
        s2 = self.posterior_mean_coef2[t].reshape(-1, 1)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t):
        if t == 0:
            return 0
        var = self.betas[t] * (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])
        return var.clip(1e-20)

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_x0 = self.reconstruct_x0(sample, t, model_output)
        mean = self.q_posterior(pred_x0, sample, t)
        if t > 0:
            noise = torch.randn_like(model_output)
            std = self.get_variance(t).sqrt()
            return mean + std * noise
        else:
            return mean

    def __len__(self):
        return self.num_timesteps


# Training & Sampling

def train_model(config):
    torch.manual_seed(42)
    dataset = dino_dataset()
    val_split = int(0.2 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_split, val_split])
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

    model = MLP(config.hidden_size, config.hidden_layers, config.embedding_size, config.time_embedding, config.input_embedding)
    noise_scheduler = NoiseScheduler(config.num_timesteps, beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses, val_losses = [], []
    frames = []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch[0]
            noise = torch.randn_like(batch)
            t = torch.randint(0, config.num_timesteps, (batch.size(0),)).long()
            noisy = noise_scheduler.add_noise(batch, noise, t)
            pred = model(noisy, t)
            loss = F.mse_loss(pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        losses.append(total_loss / len(train_loader))

        # validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch[0]
                noise = torch.randn_like(val_batch)
                t = torch.randint(0, config.num_timesteps, (val_batch.size(0),)).long()
                noisy = noise_scheduler.add_noise(val_batch, noise, t)
                pred = model(noisy, t)
                val_loss += F.mse_loss(pred, noise).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # sampling for reverse process visualization
        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            sample = torch.randn(config.eval_batch_size, 2)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for t in timesteps:
                t_tensor = torch.full((config.eval_batch_size,), t, dtype=torch.long)
                with torch.no_grad():
                    eps = model(sample, t_tensor)
                sample = noise_scheduler.step(eps, t, sample)
            frames.append(sample.numpy())

    return model, losses, val_losses, frames


def plot_losses(losses, val_losses, outdir):
    plt.figure()
    plt.plot(losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()


def save_samples(frames, outdir):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "frames.npy"), np.array(frames))
    for i, frame in enumerate(frames):
        plt.scatter(frame[:, 0], frame[:, 1], alpha=0.6)
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title(f"Epoch {i}")
        plt.savefig(os.path.join(outdir, f"sample_{i:03d}.png"))
        plt.close()

def sample_grid(model, scheduler, sample_sizes, T, title=""):
    model.eval()
    fig, axs = plt.subplots(1, len(sample_sizes), figsize=(20, 5))
    for i, n in enumerate(sample_sizes):
        x = torch.randn(n, 2)
        for t in reversed(range(T)):
            t_tensor = torch.full((n,), t, dtype=torch.long)
            with torch.no_grad():
                eps = model(x, t_tensor)
                x = scheduler.step(eps, t, x)
        axs[i].scatter(x[:, 0], x[:, 1], s=5)
        axs[i].set_title(f"{n} samples")
        axs[i].axis("equal")
        axs[i].grid(True)
        axs[i].set_xlim(-6, 6)
        axs[i].set_ylim(-6, 6)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_reverse_process(model, scheduler, T=100, n=500, save_path=None):
    model.eval()
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    x = torch.randn(n, 2)
    step_size = max(1, T // 10)
    steps = list(range(T - step_size, -1, -step_size))

    for i, t in enumerate(steps):
        x_step = x.clone()
        for j in reversed(range(t, T)):
            t_tensor = torch.full((n,), j, dtype=torch.long)
            with torch.no_grad():
                eps = model(x_step, t_tensor)
                x_step = scheduler.step(eps, j, x_step)

        ax = axs[i // 5][i % 5]
        ax.scatter(x_step[:, 0], x_step[:, 1], s=5)
        ax.set_title(f"Step {t}")
        ax.axis("equal")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.grid(True)

    plt.suptitle("Reverse Diffusion Process Visualization", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal")
    parser.add_argument("--input_embedding", type=str, default="sinusoidal")
    parser.add_argument("--save_images_step", type=int, default=10)
    config = parser.parse_args()

    model, losses, val_losses, frames = train_model(config)
    scheduler = NoiseScheduler(config.num_timesteps, beta_schedule=config.beta_schedule)

    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pth"))
    np.save(os.path.join(outdir, "loss.npy"), losses)
    np.save(os.path.join(outdir, "val_loss.npy"), val_losses)
    plot_losses(losses, val_losses, outdir)
    save_samples(frames, os.path.join(outdir, "samples"))
    sample_grid(model, scheduler, sample_sizes=[50, 200, 500, 1000], T=config.num_timesteps, title=f"Samples (T={config.num_timesteps}, {config.beta_schedule})")
    visualize_reverse_process(model, scheduler, T=100, n=500, save_path="reverse_process_grid_T100.png")

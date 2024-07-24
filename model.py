"""Defines consistency model."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def blk(ic: int, oc: int) -> nn.Module:
    return nn.Sequential(
        nn.GroupNorm(32, num_channels=ic),
        nn.SiLU(),
        nn.Conv2d(ic, oc, 3, padding=1),
        nn.GroupNorm(32, num_channels=oc),
        nn.SiLU(),
        nn.Conv2d(oc, oc, 3, padding=1),
    )


class ConsistencyModel(nn.Module):
    def __init__(self, n_channel: int, eps: float = 0.002, hdims: int = 128) -> None:
        super(ConsistencyModel, self).__init__()

        self.eps = eps

        self.freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=hdims, dtype=torch.float32) / hdims)

        self.down = nn.Sequential(
            *[
                nn.Conv2d(n_channel, hdims, 3, padding=1),
                blk(hdims, hdims),
                blk(hdims, 2 * hdims),
                blk(2 * hdims, 2 * hdims),
            ]
        )

        self.time_downs = nn.Sequential(
            nn.Linear(2 * hdims, hdims),
            nn.Linear(2 * hdims, hdims),
            nn.Linear(2 * hdims, 2 * hdims),
            nn.Linear(2 * hdims, 2 * hdims),
        )

        self.mid = blk(2 * hdims, 2 * hdims)

        self.up = nn.Sequential(
            *[
                blk(2 * hdims, 2 * hdims),
                blk(2 * 2 * hdims, hdims),
                blk(hdims, hdims),
                nn.Conv2d(2 * hdims, 2 * hdims, 3, padding=1),
            ]
        )
        self.last = nn.Conv2d(2 * hdims + n_channel, n_channel, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if isinstance(t, float):
            t = torch.tensor([t] * x.shape[0], dtype=torch.float32).to(x.device).unsqueeze(1)

        # time embedding
        args = t.float() * self.freqs[None].to(t.device)

        # so model is supposedly "time aware"
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(x.device)

        x_ori = x

        # perform F(x, t)
        hs = []
        for idx, layer in enumerate(self.down):
            if idx % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
                x = F.interpolate(x, scale_factor=0.5)
                hs.append(x)

            x = x + self.time_downs[idx](t_emb)[:, :, None, None]

        x = self.mid(x)

        for idx, layer in enumerate(self.up):
            if idx % 2 == 0:
                x = layer(x) + x
            else:
                x = torch.cat([x, hs.pop()], dim=1)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = layer(x)

        # unet concat
        x = self.last(torch.cat([x, x_ori], dim=1))

        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).pow(0.5)

        # gate: weighting between original "noisy image" and new image
        # as time progresses want to rely more on the model's output image (likely to be more informed)
        return c_skip_t[:, :, None, None] * x_ori + c_out_t[:, :, None, None] * x

    def loss(
        self,
        x: Tensor,
        z: Tensor,
        t1: Tensor,
        t2: Tensor,
        ema_model: nn.Module,
        loss_type: Literal["mse", "huber"] = "mse",
    ) -> Tensor:
        x2 = x + z * t2[:, :, None, None]

        # forward pass
        x2 = self(x2, t2)

        with torch.no_grad():
            x1 = x + z * t1[:, :, None, None]

            # exponential moving average model, shared weights as original model

            # with ema_model, want it similar to original model's so model thus
            # has consistent outputs for the same image sample for different time steps
            # across the flow.
            x1 = ema_model(x1, t1)

        match loss_type:
            case "mse":
                return F.mse_loss(x1, x2)
            case "huber":
                return pseudo_huber_loss(x1, x2)
            case _:
                raise ValueError("Invalid loss type. Choose 'mse' or 'huber'.")

    @torch.no_grad()
    def sample(self, x: Tensor, ts: list[float], partial_start: float | None = None) -> Tensor:
        if partial_start is not None:
            # Start from a partially denoised state
            start_idx = next(i for i, t in enumerate(ts) if t <= partial_start)
            x = self(x, ts[start_idx])
            ts = ts[start_idx + 1:]

        # Just running through the model at random timestamps until end
        # Bigger jumps more unstable
        for t in ts:
            z = torch.randn_like(x)
            x = x + (math.sqrt(t**2 - self.eps**2) * z)
            x = self(x, t)

        return x


def pseudo_huber_loss(x: Tensor, y: Tensor, delta: float = 1.0) -> Tensor:
    diff = x - y
    return torch.mean(delta**2 * (torch.sqrt(1 + (diff / delta) ** 2) - 1))


def kerras_boundaries(sigma: float, eps: float, n: int, t: float) -> Tensor:
    # This will be used to generate the boundaries for the time discretization
    bounds = [(eps ** (1 / sigma) + i / (n - 1) * (t ** (1 / sigma) - eps ** (1 / sigma))) ** sigma for i in range(n)]
    return torch.tensor(bounds)

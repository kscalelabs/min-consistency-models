import torch
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import math
from typing import List
import argparse
import os

from model import ConsistencyModel 
from dataloader import mnist 
from utils.kerras_boundaries import kerras_boundaries 


def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_channels = 1
    name = "mnist"

    train_loader, test_loader = mnist()
    model = ConsistencyModel(n_channels, D=128)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    ema_model = ConsistencyModel(n_channels, D=128)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.n_epochs+1):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / args.n_epochs) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device) # blackbox "time generator"

        pbar = tqdm(train_loader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            z = torch.randn_like(x) # random noise
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            loss = model.loss(x, z, t_0, t_1, ema_model=ema_model, loss_type=args.loss_type)

            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                # because model is EMA, we want to EMA the loss as well
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}

                # EMA of the model's parameters
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")

        model.eval()
        with torch.no_grad():
            # Sample 5 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, os.path.join(args.output_dir, f"{args.prefix}ct_{name}_sample_5step_{epoch}.png"))

            # Sample 2 Steps -- possible due to consistency modeling
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, os.path.join(args.output_dir, f"{args.prefix}ct_{name}_sample_2step_{epoch}.png"))

            # save model
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.prefix}ct_{name}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency Model Training")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for checkpoint and output names")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--output_dir", type=str, default="./contents", help="Output directory for checkpoints and images")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use")
    parser.add_argument("--loss_type", type=str, default="mse", choices=['mse', 'huber'], help="Type of loss function to use")
    parser.add_argument("--partial_sampling", action="store_true", help="Enable partial sampling")
    args = parser.parse_args()

    main(args)
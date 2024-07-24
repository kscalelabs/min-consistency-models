"""Defines the inference script."""

import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image

from dataloader import mnist
from model import ConsistencyModel

logger = logging.getLogger(__name__)


def get_low_quality_image(test_loader: DataLoader) -> tuple[Tensor, int]:
    """Get a single low quality image from the test loader.

    Args:
        test_loader: DataLoader for the MNIST test set

    Returns:
        A low quality (12x12) image tensor and its corresponding label
    """
    # Get a single batch from the test loader
    images, labels = next(iter(test_loader))

    # Select the first image from the batch
    image = images[0]
    label = labels[0].item()

    # Resize the image to 14x14
    low_quality_image = F.interpolate(
        image.unsqueeze(0),
        size=(14, 14),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return low_quality_image, label


def finish_low_quality_image(
    model: nn.Module,
    low_quality_image: Tensor,
    device: torch.device,
    partial_start: float = 40.0,
    steps: list[float] | None = None,
) -> Tensor:
    """Finish a low quality MNIST image using partial sampling.

    Args:
        model: The trained ConsistencyModel
        low_quality_image: A tensor of size 1x12x12
        device: The device to run the model on
        partial_start: The starting point for partial sampling (default: 40.0)
        steps: List of timesteps for sampling (if None, default steps will be used)

    Returns:
        A tensor of the finished image (28x28)
    """
    # Upscale the image to 28x28
    upscaled_image = F.interpolate(
        low_quality_image.unsqueeze(0),
        size=(28, 28),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Ensure the image is on the correct device
    upscaled_image = upscaled_image.to(device)

    # Add noise corresponding to partial_start
    noisy_image = upscaled_image + torch.randn_like(upscaled_image) * partial_start

    # If steps are not provided, use default steps
    if steps is None:
        steps = [80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0]

    # Perform partial sampling
    with torch.no_grad():
        finished_image = model.sample(noisy_image.unsqueeze(0), ts=steps, partial_start=partial_start)

    # Denormalize and clamp the image
    finished_image = (finished_image.squeeze(0) * 0.5 + 0.5).clamp(0, 1)

    return finished_image


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Consistency Model Training and Image Finishing")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for checkpoint and output names")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./contents",
        help="Output directory for checkpoints and images",
    )
    args = parser.parse_args()

    n_channels = 1
    name = "mnist"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    _, test_loader = mnist()
    model = ConsistencyModel(n_channels, hdims=128)
    model.to(device)

    # Load the trained model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f"{args.prefix}ct_{name}.pth")))
    model.eval()

    # Get a low quality image from the test set
    low_quality_image, label = get_low_quality_image(test_loader)

    # Save the low quality image
    save_image(
        low_quality_image,
        os.path.join(args.output_dir, f"{args.prefix}low_quality_image.png"),
    )

    # Finish the low quality image
    finished_image = finish_low_quality_image(model, low_quality_image, device)

    # Save the finished image
    save_image(
        finished_image,
        os.path.join(args.output_dir, f"{args.prefix}finished_image.png"),
    )

    logger.info("Processed image with label: %s", label)
    logger.info("Low quality image saved as: %slow_quality_image.png", args.prefix)
    logger.info("Finished image saved as: %sfinished_image.png", args.prefix)


if __name__ == "__main__":
    main()

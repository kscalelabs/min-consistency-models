"""Defines a simple MNIST dataloader."""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations for the images
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def mnist() -> tuple[DataLoader, DataLoader]:
    # Download and load the training data
    train_dataset = datasets.MNIST(
        root="mnist_data", train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,  # Set the batch size
        shuffle=True,  # Shuffle the dataset
    )

    # Download and load the test data
    test_dataset = datasets.MNIST(
        root="mnist_data", train=False, download=True, transform=transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1000,  # Set the batch size
        shuffle=False,  # No need to shuffle the test dataset
    )

    return train_loader, test_loader

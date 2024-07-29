"""Defines simple dataloaders."""

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist() -> Tuple[DataLoader, DataLoader]:
    # Define the transform for the images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    train_dataset = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,  # Set the batch size
        shuffle=True,  # Shuffle the dataset
    )

    # Download and load the test data
    test_dataset = datasets.MNIST(root="mnist_data", train=False, download=True, transform=transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1000,  # Set the batch size
        shuffle=False,  # No need to shuffle the test dataset
    )

    return train_loader, test_loader


def cifar() -> Tuple[DataLoader, DataLoader]:
    # Define the transforms for the images
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 3 channel
        ]
    )
    # Download and load the training data
    train_dataset = datasets.CIFAR10(root="cifar_data", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.CIFAR10(root="cifar_data", train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

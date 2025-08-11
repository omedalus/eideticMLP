import math
import torch
from torch.utils.data import DataLoader, Subset

from typing import Optional, Tuple


def load_MNIST_dataset(
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST dataset and returns DataLoader objects for training and test sets.

    The images are transformed to tensors, flattened to 1D vectors of length 784.

    Args:
        train_size (int, optional): Number of training samples to use. If None, use all.
        test_size (int, optional): Number of test samples to use. If None, use all.

    Returns:
        Tuple[DataLoader, DataLoader]:
            - train_loader: DataLoader for the training set (batch_size=64, shuffled)
            - test_loader: DataLoader for the test set (batch_size=1000, not shuffled)
    """
    from torchvision import datasets, transforms

    # Transform: convert images to tensors and normalize to [0, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.view(-1)
            ),  # Flatten to 1D tensor of shape (784,)
        ]
    )
    train_dataset = datasets.MNIST(
        root="./runtime-data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./runtime-data",
        train=False,
        download=True,
        transform=transform,
    )

    if train_size is not None:
        train_dataset = Subset(train_dataset, list(range(train_size)))
    if test_size is not None:
        test_dataset = Subset(test_dataset, list(range(test_size)))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def ascii_mnist_sample(mnistsample: Tuple[torch.Tensor, int]) -> str:
    """
    Generates a string representation of a single MNIST sample in ASCII format.

    Args:
        mnistsample: A tuple containing the image tensor and its label.
    """
    image, label = mnistsample

    PIXEL_LEVEL_CHARS = ".+*#"
    pixel_levels = [max(0, min(3, math.floor(float(x) * 4))) for x in image]
    pixel_chars = [PIXEL_LEVEL_CHARS[x] for x in pixel_levels]

    pixel_length = len(pixel_levels)
    pixel_width = math.isqrt(pixel_length)

    ascii_image = ""
    for row in range(pixel_width):
        for col in range(pixel_width):
            ascii_image += pixel_chars[row * pixel_width + col]
        ascii_image += "\n"

    ascii_image += f"Label: {label}"
    return ascii_image

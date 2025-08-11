from torch.utils.data import DataLoader
from typing import Tuple


def load_MNIST_dataset() -> Tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    # Transform: convert images to tensors and normalize to [0, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.view(-1)
            ),  # Flatten to 1D tensor of shape (784,)
            transforms.Lambda(
                lambda x: x - x.mean() + 0.5
            ),  # Normalize mean to 0.5 per sample
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
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def main():
    datatrain, datatest = load_MNIST_dataset()

    print("Training set size:", len(datatrain.dataset))
    print("Test set size:", len(datatest.dataset))

    print("Single element size:", datatrain.dataset[0][0].size())


if __name__ == "__main__":
    main()

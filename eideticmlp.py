import torch
import torch.nn as nn
import torch.optim as optim
import _mnist_helpers

import _mlp_conventional_topologies
import _mlp_novel_topology

NUM_EPOCHS = 30


def train_mlp(train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _mlp_novel_topology.MLP_2HLSkipWithEideticMem().to(device)
    # model = _mlp_conventional_topologies.MLP_2HLSkip().to(device)

    model.eidetic_mem.enabled = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    last_nummemkeys = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)

        nummemkeys = len(model.eidetic_mem._storage.keys())

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Test Accuracy: {acc:.2f}% -- Eidetic memory cells: {nummemkeys} (+{nummemkeys - last_nummemkeys})"
        )
        last_nummemkeys = nummemkeys

    # print(model.eidetic_mem.diagnostic_print())


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    datatrain, datatest = _mnist_helpers.load_MNIST_dataset(
        train_size=1000,
        test_size=100,
    )

    print("Training set size:", len(datatrain.dataset))
    print("Test set size:", len(datatest.dataset))

    print("Single element size:", datatrain.dataset[0][0].size())
    # print(_mnist_helpers.ascii_mnist_sample(datatrain.dataset[2]))

    print("\nStarting training...")
    train_mlp(datatrain, datatest)


if __name__ == "__main__":
    main()

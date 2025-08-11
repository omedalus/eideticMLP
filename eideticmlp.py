import torch
import torch.nn as nn
import torch.optim as optim
import _mnist_helpers

NUM_EPOCHS = 10


# Standard MLP for MNIST (input 784, two hidden layers, output 10)
class MLP_2HLStandard(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


# MLP with skip connections
class MLP_2HLSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_skip = nn.Linear(
            784, 32
        )  # skip connection from input to 2nd hidden layer
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1) + self.fc_skip(x))
        out = self.fc3(h2)
        return out


def train_mlp(train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MLP_2HLStandard().to(device)
    model = MLP_2HLSkip().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} - Test Accuracy: {acc:.2f}%"
        )


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
    datatrain, datatest = _mnist_helpers.load_MNIST_dataset()

    print("Training set size:", len(datatrain.dataset))
    print("Test set size:", len(datatest.dataset))

    print("Single element size:", datatrain.dataset[0][0].size())
    # print(_mnist_helpers.ascii_mnist_sample(datatrain.dataset[2]))

    print("\nStarting training...")
    train_mlp(datatrain, datatest)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import _mnist_helpers

from eidetic_hidden_layer_lookup import EideticHiddenLayerLookup

NUM_EPOCHS = 10


# MLP with skip connections and a memory injector.
class MLP_2HLSkipWithEideticMem(nn.Module):
    def __init__(self):
        super().__init__()
        self.eidetic_mem = EideticHiddenLayerLookup()

        self.fullconn_sensory_to_indexer = nn.Linear(784, 64)
        self.fullconn_indexer_to_integrator = nn.Linear(64, 32)

        self.fullconn_sensory_skip_to_integrator = nn.Linear(784, 32)
        self.fullconn_recaller_to_integrator = nn.Linear(784, 32)

        self.fullconn_integrator_to_output = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x_sensory: torch.Tensor):
        # Remember: x_sensory is a *batch* of sensory inputs.
        activations_indexer = self.relu(self.fullconn_sensory_to_indexer(x_sensory))

        # activations_indexer is now a *batch* of activation vectors of the
        # indexer layer. For each one, we need to find the corresponding
        # past sensory vector in the eidetic memory, which will populate
        # the recall layer.
        x_recaller = torch.zeros_like(x_sensory)
        if len(self.eidetic_mem) > 0:
            x_recaller = self.eidetic_mem.lookup_batch(activations_indexer)

        # Update the eidetic memory, associating the current
        # indexer activations with the sensory input
        self.eidetic_mem.insert_batch(activations_indexer, x_sensory)

        activations_integrator = self.relu(
            self.fullconn_indexer_to_integrator(activations_indexer)
            + self.fullconn_recaller_to_integrator(x_recaller)
        )

        activations_output = self.fullconn_integrator_to_output(activations_integrator)
        return activations_output


def train_mlp(train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP_2HLSkipWithEideticMem().to(device)

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

import _mnist_helpers


def main():
    datatrain, datatest = _mnist_helpers.load_MNIST_dataset()

    print("Training set size:", len(datatrain.dataset))
    print("Test set size:", len(datatest.dataset))

    print("Single element size:", datatrain.dataset[0][0].size())
    print(_mnist_helpers.ascii_mnist_sample(datatrain.dataset[2]))


if __name__ == "__main__":
    main()

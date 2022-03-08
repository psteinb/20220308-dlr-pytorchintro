import torch
import torchvision
from torchvision import datasets, transforms


def main(somepath="./pytorch-data"):

    transforms_ = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    raw_dataset = datasets.MNIST(somepath, download=True, transform=transforms_)

    img, label = raw_dataset[0]

    assert isinstance(img, torch.Tensor)
    assert img.mean() < 1.0
    assert img.mean() > 0.0
    print("first image in normalized MNIST:", img.mean(), img.std())
    # print("normalized batch mean value (fp32):", nbatch.float().mean())
    # print("normalized batch std  value (fp32):", torch.std(nbatch.float()))

    train_dataset = datasets.MNIST(
        somepath, download=True, transform=transforms_, train=True
    )
    test_dataset = datasets.MNIST(
        somepath, download=True, transform=transforms_, train=False
    )
    assert len(train_dataset) > len(test_dataset)
    assert len(train_dataset) == len(raw_dataset)
    print(len(train_dataset), len(test_dataset))

    ## cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
        # **cuda_kwargs
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    print("our data loaders for MNIST")
    print(train_dataset)
    print(train_loader)

    for batch_idx, (X, y) in enumerate(train_loader):
        print(batch_idx, X.shape, y.shape)
        break


if __name__ == "__main__":
    main()

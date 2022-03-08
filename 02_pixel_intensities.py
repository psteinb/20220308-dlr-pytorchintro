import torch
import torchvision
from torchvision import datasets


def main(somepath="./pytorch-data"):

    raw_dataset = datasets.MNIST(somepath, download=True)

    imglist = []
    lbllist = []
    for i in range(32):
        img, lbl = raw_dataset[i]
        imglist.append(img)
        lbllist.append(lbl)

    trf = torchvision.transforms.PILToTensor()
    timages = []
    for img in imglist:
        timages.append(trf(img))

    batch = torch.concat(timages)
    assert len(batch.shape) == 3
    assert batch.shape == (32, 28, 28)

    # print("batch mean value (raw) :", batch.mean())
    print("batch mean value (fp32):", batch.float().mean())
    print("batch std  value (fp32):", torch.std(batch.float()))
    # more information can be obtained from the API documentation:
    # https://pytorch.org/docs/stable/tensors.html

    nbatch = batch / 255.0
    print("normalized batch mean value (fp32):", nbatch.float().mean())
    print("normalized batch std  value (fp32):", torch.std(nbatch.float()))


if __name__ == "__main__":
    main()

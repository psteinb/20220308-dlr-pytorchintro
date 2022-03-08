import torch
import torchvision
from torchvision import datasets


def main(somepath="./pytorch-data"):

    raw_dataset = datasets.MNIST(somepath, download=True)

    assert len(raw_dataset) > 0
    assert len(raw_dataset) == 60_000

    asample = raw_dataset[0]
    assert len(asample) > 0
    assert len(asample) == 2

    image, label = asample
    # PIL.Image.Image
    assert "Image" in type(image).__name__
    assert image.width > 0
    assert image.height > 0
    assert image.width == 28
    assert image.height == 28

    assert image.getextrema() == (0, 255)
    assert len(image.getbands()) == 1
    assert image.getbands() == ("L",)

    assert label == 5

    # to display
    # > display(image)
    # or
    # > %matplotlib inline
    # > from matplotlib.pyplot import imshow
    # > imshow(image)

    trf = torchvision.transforms.PILToTensor()
    timage = trf(image)

    assert not "Image" in type(timage).__name__
    assert timage.shape
    assert len(timage.shape) == 3
    assert timage.shape == (1, 28, 28)


if __name__ == "__main__":
    main()

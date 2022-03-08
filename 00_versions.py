import torch
import torchvision


def main():
    print(torch.__version__)
    print(torchvision.__version__)

    print("cuda devices available", torch.cuda.device_count())


if __name__ == "__main__":
    main()

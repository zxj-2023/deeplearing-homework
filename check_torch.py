import torch


def main():
    print("torch_version:", torch.__version__)
    print("cuda_version:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()

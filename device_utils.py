import torch


def get_preferred_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return torch.device("cpu")

    # The installed PyTorch build only supports up to sm_90.
    # Fall back to CPU on newer GPUs instead of failing during CUDA tensor creation.
    if major > 9:
        return torch.device("cpu")

    return torch.device("cuda")
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    自动选择运行设备。

    MacBook M 系列优先使用 mps。
    如果 mps 不可用，就使用 cpu。
    如果是 NVIDIA 显卡环境，就使用 cuda。
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def rmse_loss(pred, target):
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    return rmse
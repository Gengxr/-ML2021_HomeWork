import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class CovidDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        else:
            return self.x[index]


def load_train_valid_data(train_path, valid_ratio=0.2, seed=42):
    """
    读取训练集，并划分为训练集和验证集。

    数据格式：
    第 1 列：id，不参与训练
    中间列：输入特征
    最后 1 列：tested_positive，作为预测目标
    """

    df = pd.read_csv(train_path)

    # 去掉第 1 列 id，最后 1 列作为标签
    x = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    # 打乱数据
    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    # 划分训练集和验证集
    valid_size = int(len(x) * valid_ratio)

    x_valid = x[:valid_size]
    y_valid = y[:valid_size]

    x_train = x[valid_size:]
    y_train = y[valid_size:]

    # 标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    return x_train, y_train, x_valid, y_valid, scaler


def load_test_data(test_path, scaler):
    """
    读取测试集。

    数据格式：
    第 1 列：id
    后面列：输入特征
    """

    df = pd.read_csv(test_path)

    # 保存 id，后面生成 submission.csv 时可以使用
    ids = df.iloc[:, 0].values

    # 去掉第 1 列 id
    x_test = df.iloc[:, 1:].values

    # 使用训练集的 scaler 做标准化
    x_test = scaler.transform(x_test)

    return ids, x_test
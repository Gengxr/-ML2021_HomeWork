import os
import json
import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CovidDataset, load_test_data
from model import RegressionModel
from utils import get_device


def predict():
    test_path = "data/covid.test.csv"
    sample_path = "data/sampleSubmission.csv"

    model_path = "checkpoints/best_model.pth"
    scaler_path = "checkpoints/scaler.pkl"
    config_path = "checkpoints/config.json"

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    # 读取模型配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    input_dim = config["input_dim"]

    # 读取标准化工具
    scaler = joblib.load(scaler_path)

    # 加载测试集
    ids, x_test = load_test_data(test_path, scaler)

    test_dataset = CovidDataset(x_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # 创建模型并加载参数
    model = RegressionModel(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)

            pred = model(batch_x)
            pred = pred.cpu().numpy()

            predictions.extend(pred.tolist())

    # 读取 sampleSubmission，保持 Kaggle 要求的格式
    submission = pd.read_csv(sample_path)

    # 通常第 1 列是 id，第 2 列是 tested_positive
    submission.iloc[:, 1] = predictions

    save_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(save_path, index=False)

    print("Submission saved to:", save_path)


if __name__ == "__main__":
    predict()
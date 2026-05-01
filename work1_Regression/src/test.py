import json
import joblib
import torch
from torch.utils.data import DataLoader

from dataset import CovidDataset, load_train_valid_data
from model import RegressionModel
from utils import get_device, rmse_loss


def test():
    # 路径设置
    train_path = "data/covid.train.csv"

    model_path = "checkpoints/best_model.pth"
    scaler_path = "checkpoints/scaler.pkl"
    config_path = "checkpoints/config.json"

    batch_size = 64
    valid_ratio = 0.2
    seed = 42

    # 选择设备
    device = get_device()
    print("Using device:", device)

    # 读取模型配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    input_dim = config["input_dim"]
    print("Input dim:", input_dim)

    # 重新读取训练集，并划分出验证集
    # 注意：这里的划分方式要和 train.py 保持一致，seed 也要一致
    x_train, y_train, x_valid, y_valid, scaler = load_train_valid_data(
        train_path=train_path,
        valid_ratio=valid_ratio,
        seed=seed
    )

    print("Valid size:", len(x_valid))

    # 构建验证集 Dataset 和 DataLoader
    valid_dataset = CovidDataset(x_valid, y_valid)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 创建模型
    model = RegressionModel(input_dim=input_dim).to(device)

    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 开始测试
    total_rmse = 0.0
    total_mse = 0.0
    total_samples = 0

    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)

            rmse = rmse_loss(pred, batch_y)
            mse = torch.mean((pred - batch_y) ** 2)

            batch_size_now = batch_x.size(0)

            total_rmse += rmse.item() * batch_size_now
            total_mse += mse.item() * batch_size_now
            total_samples += batch_size_now

            predictions.extend(pred.cpu().numpy().tolist())
            targets.extend(batch_y.cpu().numpy().tolist())

    avg_rmse = total_rmse / total_samples
    avg_mse = total_mse / total_samples

    print("Test on validation set finished.")
    print(f"MSE:  {avg_mse:.6f}")
    print(f"RMSE: {avg_rmse:.6f}")

    # 打印前 10 条预测结果和真实结果对比
    print("\n前 10 条预测结果对比：")
    print("index\tprediction\ttarget")
    for i in range(min(10, len(predictions))):
        print(f"{i}\t{predictions[i]:.6f}\t{targets[i]:.6f}")


if __name__ == "__main__":
    test()
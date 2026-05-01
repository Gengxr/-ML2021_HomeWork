import os
import json
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CovidDataset, load_train_valid_data
from model import RegressionModel
from utils import set_seed, get_device, rmse_loss


def train():
    set_seed(42)

    # 路径设置
    train_path = "data/covid.train.csv"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 超参数设置
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 300
    valid_ratio = 0.2

    # 设备选择：MacBook M 系列一般会使用 mps
    device = get_device()
    print("Using device:", device)

    # 读取并处理数据
    x_train, y_train, x_valid, y_valid, scaler = load_train_valid_data(
        train_path=train_path,
        valid_ratio=valid_ratio,
        seed=42,
    )
 
    input_dim = x_train.shape[1]

    print("Input dim:", input_dim)
    print("Train size:", len(x_train))
    print("Valid size:", len(x_valid))

    # 构建 Dataset
    train_dataset = CovidDataset(x_train, y_train)
    valid_dataset = CovidDataset(x_valid, y_valid)

    # 构建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 创建模型
    model = RegressionModel(input_dim=input_dim).to(device)

    # 回归任务常用 MSELoss
    criterion = nn.MSELoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_valid_rmse = float("inf")

    # 开始训练
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 1. 前向传播
            pred = model(batch_x)

            # 2. 计算损失
            loss = criterion(pred, batch_y)

            # 3. 清空梯度
            optimizer.zero_grad()

            # 4. 反向传播
            loss.backward()

            # 5. 更新参数
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss = train_loss / len(train_dataset)

        # 验证模型
        model.eval()
        valid_rmse_total = 0.0

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred = model(batch_x)
                rmse = rmse_loss(pred, batch_y)

                valid_rmse_total += rmse.item() * batch_x.size(0)

        valid_rmse = valid_rmse_total / len(valid_dataset)

        # 保存验证集上效果最好的模型
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse

            model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)

            scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)

            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({"input_dim": input_dim}, f)

        # 每 10 轮打印一次结果
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.6f} "
                f"Valid RMSE: {valid_rmse:.6f} "
                f"Best RMSE: {best_valid_rmse:.6f}"
            )

    print("Training finished.")
    print("Best valid RMSE:", best_valid_rmse)


if __name__ == "__main__":
    train()
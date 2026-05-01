# ML2021 Spring HW1 - Regression

本项目用于完成 Kaggle Competition [ml2021spring-hw1](https://www.kaggle.com/c/ml2021spring-hw1)：根据前三天的 COVID-19 调查数据预测第三天的 tested positive 数值。

## 目录结构

```text
work1_Regression/
├── README.md
├── requirements.txt
├── data/
│   ├── covid.test.csv
│   ├── covid.train.csv
│   └── sampleSubmission.csv
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── checkpoints/
└── outputs/
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 下载数据

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

然后下载比赛数据：

```bash
kaggle competitions download -c ml2021spring-hw1 -p data --unzip
```

确认 `data/` 下存在：

```text
covid.train.csv
covid.test.csv
sampleSubmission.csv
```

## 训练

默认使用适合本作业的特征组合：40 个州 one-hot、三天的 COVID-like illness 特征、前两天的 `tested_positive`。

```bash
conda run -n ml2021_homework python src/train.py
```

常用参数：

```bash
conda run -n ml2021_homework python src/train.py \
  --feature-mode strong \
  --epochs 3000 \
  --batch-size 270 \
  --lr 0.001 \
  --hidden-dims 16 32 \
  --l1-weight 0.00001
```

训练完成后会保存最佳模型：

```text
checkpoints/best_model.pt
```

## 预测并生成提交文件

```bash
conda run -n ml2021_homework python src/predict.py
```

输出文件：

```text
outputs/submission.csv
```

可以把该文件上传到 Kaggle。

## 文件说明

- `src/dataset.py`: 读取 CSV、切分 train/dev、特征选择和标准化。
- `src/model.py`: PyTorch MLP 回归模型。
- `src/train.py`: 训练、验证、early stopping、保存 checkpoint。
- `src/predict.py`: 加载 checkpoint 并生成 Kaggle submission。
- `src/utils.py`: 随机种子、RMSE、checkpoint、目录检查等工具函数。

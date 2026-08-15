# MNIST PyTorch Classification

基于 PyTorch 实现的 MNIST 手写数字分类项目。

这个项目主要用于练习一个完整深度学习项目的工程结构，并在此基础上逐步改进模型和训练策略，提高 MNIST 分类准确率。

## 项目结构

```text
MNIST/
│
├── data/                  # MNIST 数据集
├── checkpoints/           # 模型权重与训练状态
├── logs_train/            # TensorBoard 日志
│
├── models/
│   ├── __init__.py
│   └── basic_cnn.py       # BasicCNN
│
├── utils/
│   ├── __init__.py
│   └── checkpoint.py      # Checkpoint 保存与加载
│
├── config.py              # 超参数与路径配置
├── dataset.py             # 数据集与 DataLoader
├── train.py               # 训练
├── test.py                # 测试
├── main.py                # 程序入口
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 当前进度

目前已经完成：

* [x] MNIST 数据集与 DataLoader
* [x] BasicCNN 模型
* [x] 模型训练与测试
* [x] Loss / Accuracy 统计
* [x] TensorBoard 日志记录
* [x] `config.py` 统一管理配置
* [x] 项目代码模块化拆分
* [x] Checkpoint 保存与加载
* [x] 支持断点续训
* [x] `.gitignore`
* [x] `requirements.txt`
* [x] `README.md`

目前正在改进：

* [ ] 使用 SmallResNet 替换 BasicCNN
* [ ] 调整模型结构提高准确率
* [ ] 调整 Epoch、Learning Rate、Batch Size 等超参数
* [ ] 加入 Learning Rate Scheduler

## 后续计划

后续准备逐步加入：

* [ ] 固定随机种子
* [ ] 更完善的学习率调度策略
* [ ] 数据增强
* [ ] Train / Validation / Test 正式划分
* [ ] Best Model 保存
* [ ] 不同模型与超参数实验对比
* [ ] `argparse` 命令行参数
* [ ] 更完善的实验管理

## 运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行项目：

```bash
python main.py
```

查看 TensorBoard：

```bash
tensorboard --logdir=logs_train
```

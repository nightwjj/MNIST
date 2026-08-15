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
│   ├── basic_cnn.py       # BasicCNN
│   ├── m3_cnn.py          # M3CNN
│   └── resnet.py          # SmallResNet
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py      # Checkpoint 保存与加载
│   └── seed.py            # 固定随机种子
│
├── config.py              # 超参数与路径配置
├── dataset.py             # 数据集与 DataLoader
├── train.py               # 训练
├── test.py                # 测试
├── mnist.py               # 训练程序入口
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 当前进度

目前已经完成：

* [x] MNIST 数据集、DataLoader 与基础数据增强
* [x] BasicCNN、M3CNN 和 SmallResNet 模型
* [x] 模型训练与测试
* [x] Loss / Accuracy 统计
* [x] TensorBoard 日志记录
* [x] `config.py` 统一管理配置
* [x] 项目代码模块化拆分
* [x] Checkpoint 保存与加载基础功能
* [x] 根据测试集准确率保存最佳模型
* [x] StepLR 学习率调度器
* [x] 固定随机种子
* [x] `.gitignore`
* [x] `requirements.txt`
* [x] `README.md`

目前正在进行：

* [ ] 完善断点续训流程
* [ ] 对比 BasicCNN、M3CNN 和 SmallResNet
* [ ] 调整模型结构提高准确率
* [ ] 调整 Epoch、Learning Rate、Batch Size 等超参数

## 后续计划

后续准备逐步加入：

* [ ] 更完善的学习率调度策略
* [ ] Train / Validation / Test 正式划分
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
python mnist.py
```

查看 TensorBoard：

```bash
tensorboard --logdir=logs_train
```

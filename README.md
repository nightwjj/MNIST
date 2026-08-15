# MNIST PyTorch Classification

基于 PyTorch 实现的 MNIST 手写数字分类项目。

这个项目主要用于学习一个完整深度学习项目的开发过程，包括数据处理、模型构建、训练与验证、实验记录、模型保存以及使用训练好的模型预测自己的手写数字图片。

## 项目结构

```text
MNIST/
├── data/                       # MNIST 数据集（运行后生成）
├── outputs/                    # 不同实验的输出（运行后生成）
│   └── basic_cnn_exp04/
│       ├── checkpoints/
│       │   └── best.pth        # 验证集准确率最高的模型
│       ├── tensorboard/        # TensorBoard 日志
│       ├── config.json         # 本次实验配置
│       └── results.json        # 本次实验最终结果
│
├── models/
│   ├── __init__.py
│   ├── basic_cnn.py            # BasicCNN
│   ├── build.py                # 模型工厂
│   ├── m3_cnn.py               # M3CNN
│   └── resnet.py               # SmallResNet
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py           # 模型 checkpoint 保存与加载
│   ├── experiment.py           # 实验配置与结果保存
│   └── seed.py                 # 固定随机种子
│
├── arguments.py                # 训练命令行参数
├── config.py                   # 模型、超参数与实验路径配置
├── dataset.py                  # 数据集、数据增强与 DataLoader
├── train.py                    # 单轮训练与模型评估
├── mnist.py                    # 训练程序入口
├── test.py                     # 独立测试入口
├── predict.py                  # 单张手写数字图片预测
├── requirements.txt
├── .gitignore
└── README.md
```

## 已完成功能

- [x] MNIST 数据集与 DataLoader
- [x] Train / Validation / Test 正式划分
- [x] RandomAffine 基础数据增强
- [x] BasicCNN、M3CNN 和 SmallResNet
- [x] 模型工厂，根据名称创建模型
- [x] 训练集与验证集 Loss / Accuracy 统计
- [x] 根据验证集准确率保存最佳模型
- [x] 独立测试最佳模型
- [x] StepLR 学习率调度器
- [x] 固定随机种子
- [x] TensorBoard 日志
- [x] `argparse` 训练轮数参数
- [x] 按实验名称管理输出目录
- [x] 保存 `config.json` 和 `results.json`
- [x] 使用训练好的模型预测单张 PNG/JPG 图片

## 环境安装

建议先进入自己的 PyTorch 环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

## 配置实验

训练前可以在 `config.py` 中选择模型和实验名称：

```python
MODEL_NAME = "basic_cnn"
EXPERIMENT_TAG = "exp04"
```

支持的模型名称：

```text
basic_cnn
m3_cnn
resnet
```

每次开始新实验时，建议修改 `EXPERIMENT_TAG`，避免覆盖之前的实验结果。

## 训练模型

使用 `config.py` 中的默认训练轮数：

```bash
python mnist.py
```

通过命令行指定训练轮数：

```bash
python mnist.py --epochs 5
```

训练过程中会在验证集上评估模型，并将验证集准确率最高的 checkpoint 保存到当前实验目录。

## 测试最佳模型

```bash
python test.py
```

程序会根据 `config.py` 中的当前模型和实验目录加载 `best.pth`，然后在测试集上进行评估。

## 预测自己的手写数字图片

### 1. 准备图片

可以使用 Windows 画图等工具制作一张手写数字图片：

- 使用白色背景和黑色粗笔画；
- 一张图片只写一个数字；
- 数字尽量写在中间，四周保留空白；
- 保存为 PNG 或 JPG，例如 `7.png`。

图片不必提前调整成 28×28。`predict.py` 会自动完成灰度化、黑白反转、数字区域裁剪、缩放、居中和归一化。

### 2. 运行预测

```bash
python predict.py --image "./7.png"
```

也可以使用图片的绝对路径：

```bash
python predict.py --image "D:\path\to\7.png"
```

默认使用 `config.py` 当前设置的模型名称和最佳 checkpoint。也可以手动指定：

```bash
python predict.py --image "./7.png" --model basic_cnn --checkpoint "./outputs/basic_cnn_exp04/checkpoints/best.pth"
```

预测结果示例：

```text
加载模型：basic_cnn
模型来自第 10 轮
保存时验证集准确率：0.9920
预测结果：7
置信度：100.00%
概率最高的三个数字：
  数字 7：100.00%
  数字 9：0.00%
  数字 2：0.00%
```

## 查看 TensorBoard

查看所有实验：

```bash
tensorboard --logdir=./outputs
```

启动后，根据终端显示的地址在浏览器中打开 TensorBoard。

## 后续学习计划

- [ ] 对比 BasicCNN、M3CNN 和 SmallResNet
- [ ] 生成不同实验的结果对比表
- [ ] 混淆矩阵与每个数字类别的准确率
- [ ] 错误样本可视化
- [ ] 调整学习率、Batch Size 和模型结构
- [ ] 增加更完整的命令行参数

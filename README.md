# MNIST CNN

使用 PyTorch 实现的 MNIST 手写数字分类项目。

项目主要用于学习：

- PyTorch 基础训练流程
- CNN 模型
- DataLoader
- TensorBoard
- 模型保存与加载
- 深度学习项目工程化结构

## Project Structure

```text
MNIST_CNN/
│
├── data/
│
├── checkpoints/
├── logs_train/
│
├── models/
│   ├── __init__.py
│   └── basic_cnn.py
│
├── utils/
│   ├── __init__.py
│   └── checkpoint.py
│
├── config.py
├── dataset.py
├── train.py
├── test.py
├── main.py
│
├── requirements.txt
└── README.md
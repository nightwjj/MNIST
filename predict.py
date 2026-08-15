import argparse
import os

import torch
from PIL import Image, ImageOps
from torchvision import transforms

import config

from models.build import build_model
from utils.checkpoint import load_checkpoint


IMAGE_SIZE = 28
IMAGE_MEAN = (0.5,)
IMAGE_STD = (0.5,)


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用训练好的模型预测手写数字图片"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="待预测图片的路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        help=f"模型名称，默认值为 {config.MODEL_NAME}",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=config.BEST_MODEL_PATH,
        help=f"模型权重路径，默认值为 {config.BEST_MODEL_PATH}",
    )

    return parser.parse_args()


def load_model(model_name, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"没有找到模型权重：{checkpoint_path}"
        )

    model = build_model(model_name).to(device)
    epoch, accuracy = load_checkpoint(
        model=model,
        optimizer=None,
        path=checkpoint_path,
        device=device,
    )

    # BatchNorm 和 Dropout 在预测时必须使用评估模式。
    model.eval()

    return model, epoch, accuracy


def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"没有找到待预测图片：{image_path}"
        )

    with Image.open(image_path) as source_image:
        image = source_image.convert("L")

    # MNIST 是黑底白字，普通手写图片通常是白底黑字。
    width, height = image.size
    corner_pixels = (
        image.getpixel((0, 0)),
        image.getpixel((width - 1, 0)),
        image.getpixel((0, height - 1)),
        image.getpixel((width - 1, height - 1)),
    )
    if sum(corner_pixels) / len(corner_pixels) > 127:
        image = ImageOps.invert(image)

    image = ImageOps.autocontrast(image)

    # 裁出数字主体，再按 MNIST 的样式放入 28×28 黑色画布。
    foreground = image.point(
        lambda pixel: 255 if pixel > 30 else 0
    )
    bounding_box = foreground.getbbox()
    if bounding_box is None:
        raise ValueError("图片中没有检测到数字")

    digit = image.crop(bounding_box)
    digit_width, digit_height = digit.size
    scale = 20 / max(digit_width, digit_height)
    resized_size = (
        max(1, round(digit_width * scale)),
        max(1, round(digit_height * scale)),
    )
    digit = digit.resize(
        resized_size,
        Image.Resampling.LANCZOS,
    )

    prepared_image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=0,
    )
    paste_position = (
        (IMAGE_SIZE - resized_size[0]) // 2,
        (IMAGE_SIZE - resized_size[1]) // 2,
    )
    prepared_image.paste(digit, paste_position)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # 必须与 dataset.py 中训练和测试时的归一化保持一致。
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])

    # 增加 batch 维度：[1, 28, 28] -> [1, 1, 28, 28]。
    return transform(prepared_image).unsqueeze(0)


@torch.inference_mode()
def predict(model, image_tensor, device):
    logits = model(image_tensor.to(device))
    probabilities = torch.softmax(logits, dim=1)[0]

    confidence, predicted_digit = probabilities.max(dim=0)
    top_probabilities, top_digits = torch.topk(probabilities, k=3)

    return (
        int(predicted_digit.item()),
        float(confidence.item()),
        top_digits.cpu().tolist(),
        top_probabilities.cpu().tolist(),
    )


def main():
    args = parse_args()

    try:
        model, epoch, saved_accuracy = load_model(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            device=config.DEVICE,
        )
        image_tensor = preprocess_image(args.image)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"预测失败：{error}") from error

    (
        predicted_digit,
        confidence,
        top_digits,
        top_probabilities,
    ) = predict(
        model=model,
        image_tensor=image_tensor,
        device=config.DEVICE,
    )

    print(f"加载模型：{args.model}")
    print(f"模型权重：{args.checkpoint}")
    print(f"模型来自第 {epoch} 轮")
    print(f"保存时验证集准确率：{float(saved_accuracy):.4f}")
    print(f"预测结果：{predicted_digit}")
    print(f"置信度：{confidence:.2%}")
    print("概率最高的三个数字：")

    for digit, probability in zip(
        top_digits,
        top_probabilities,
    ):
        print(f"  数字 {digit}：{probability:.2%}")


if __name__ == "__main__":
    main()

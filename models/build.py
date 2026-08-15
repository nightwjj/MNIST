from models.basic_cnn import Basic_cnn
from models.m3_cnn import M3CNN
from models.resnet import SmallResNet

MODEL_REGISTRY = {
    "basic_cnn": Basic_cnn,
    "m3_cnn": M3CNN,
    "resnet": SmallResNet,
}

def build_model(model_name):
    if model_name not in MODEL_REGISTRY:
        supported_models = ", ".join(MODEL_REGISTRY.keys())

        raise ValueError(
            f"不支持的模型：{model_name}。"
            f"可用模型：{supported_models}"
        )

    model_class = MODEL_REGISTRY[model_name]

    return model_class()
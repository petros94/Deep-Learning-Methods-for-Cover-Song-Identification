import json
from models.resnet18.ResNet18 import from_config as make_resnet
from models.cnn.cnn import from_config as make_cnn
from models.lstm.lstm import from_config as make_lstm
from models.vit.vit import from_config as make_vit
from models.lstm_cnn.lstm_cnn import from_config as make_lstm_cnn
from models.efficientnet.EfficientNet import from_config as make_efficientnet
from models.embeddings.embeddings import from_config as make_embeddings
from models.mert_v0.mert import from_config as make_mert
import torch
from utils.generic import get_device


def make_model(config_path: str = "models/config.json"):
    """Factory to generate models from configuration

    Args:
        type (str): the model type. Check available_types above
        config_path (str): the configuration path

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    with open(config_path, "r") as f:
        config = json.load(f)


    if config["model"]["type"] == "resnet18":
        model = make_resnet(config_path=config["model"]["config_path"])
    elif config["model"]["type"] == "cnn":
        model = make_cnn(config_path=config["model"]["config_path"])
    elif config["model"]["type"] == "lstm":
        model = make_lstm(config_path=config["model"]["config_path"])
    elif config["model"]["type"] == "lstm+cnn":
        model = make_lstm_cnn(config_path=config["model"]["config_path"])
    elif config["model"]["type"] == "vit":
        model = make_vit(config_path=config["model"]["config_path"])
    elif config["model"]["type"] == "efficientnet":
        model = make_efficientnet(config_path=config["model"]["config_path"])
    elif config["model"]["type"] == "embeddings":
        model = make_embeddings(config_path=config["model"]["config_path"])
    elif config["model"]["type"] == "mert":
        model = make_mert(config_path=config["model"]["config_path"])
    else:
        raise ValueError("Invalid type")

    if config["model"]["checkpoint_path"] is not None:
        loc = get_device()
        chk = torch.load(
            config["model"]["checkpoint_path"], map_location=torch.device(loc)
        )
        print("loaded pretrained model")

        model.load_state_dict(chk["model_state_dict"])
        epoch = chk["epoch"]
        loss = chk["loss"]
        model.eval()

    return model

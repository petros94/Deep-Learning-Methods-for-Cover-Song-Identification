from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
import torch
from torch import nn
import json
from torch.nn import functional as F


class ViT(nn.Module):
    def __init__(self, patch_size=4, image_size=128) -> None:
        super().__init__()
        self.configuration = ViTConfig(
            num_channels=1, patch_size=patch_size, image_size=image_size
        )
        self.model = ViTModel(self.configuration)
        self.image_size = image_size

    def forward(self, x):
        x = F.interpolate(x, size=self.image_size)
        inputs = {"pixel_values": x}
        outputs = self.model(**inputs)
        outputs = outputs.last_hidden_state[:, 0, :].squeeze(1)
        return outputs


def from_config(config_path: str):
    """Create a ViT model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return ViT(patch_size=config["patch_size"], image_size=config["image_size"])

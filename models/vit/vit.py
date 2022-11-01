from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
import torch
from torch import nn
import json
from torch.nn import functional as F


class ViT(nn.Module):
    def __init__(
        self,
        patch_size=4,
        image_size=(12, 320),
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_size=768,
    ) -> None:
        super().__init__()
        self.configuration = ViTConfig(
            num_channels=1,
            patch_size=patch_size,
            image_size=image_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
        )
        self.model = ViTModel(self.configuration)
        self.image_size = image_size
        self.fc = nn.Linear(in_features=hidden_size, out_features=128)

    def forward(self, x):
        x = F.interpolate(x, size=self.image_size)
        inputs = {"pixel_values": x}
        outputs = self.model(**inputs)
        outputs = outputs.pooler_output
        return self.fc(outputs)


def from_config(config_path: str):
    """Create a ViT model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return ViT(
            patch_size=config["patch_size"],
            image_size=config["image_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            hidden_size=config["hidden_size"],
        )

from transformers import CvtFeatureExtractor, CvtModel, CvtConfig
import torch
from torch import nn
import json
from torch.nn import functional as F


class ViT(nn.Module):
    def __init__(
        self,
        patch_sizes=[3,3,3],
        embed_dim=[64, 192, 384],
        num_heads = [1, 3, 6],
        depth = [1, 2, 10],
        patch_stride = [1, 2, 2],
        patch_padding = [1, 1, 1]
    ) -> None:
        super().__init__()
        self.configuration = CvtConfig(
            num_channels=1,
            patch_sizes=patch_sizes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
        )
        self.model = CvtModel(self.configuration)

    def forward(self, x):
        print(x.size())
        inputs = {"pixel_values": x}
        outputs = self.model(**inputs)
        outputs = outputs.cls_token_value.squeeze(0)
        return nn.functional.normalize(outputs)


def from_config(config_path: str):
    """Create a ViT model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return ViT(
            patch_sizes=config['patch_sizes'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            depth=config['depth'],
            patch_stride=config['patch_stride'],
            patch_padding=config['patch_padding']
        )

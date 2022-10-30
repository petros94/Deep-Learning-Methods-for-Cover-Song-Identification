from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
import torch
from torch import nn
import json

class ViT(nn.Module):
    def __init__(self, patch_size=4) -> None:
        super().__init__()
        self.configuration = ViTConfig(num_channels=1, patch_size=patch_size)
        self.model = ViTModel(self.configuration)
        self.feature_extractor = ViTFeatureExtractor(do_resize=False, do_normalize=False)
        
    def forward(self, x):
        inputs = self.feature_extractor(x, return_tensors="pt")
        return self.model(**inputs)
    
def from_config(config_path: str):
    """Create a ViT model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return ViT(patch_size=config['patch_size'])
        
import torch
import torch.nn as nn
import json

from utils.generic import get_device

device = get_device()

class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.linear_1 = nn.Linear(config["input_size"], config["hidden_size"]).to(device)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config["hidden_size"], config["hidden_size"]).to(device)

    def forward(self, x):
        output = self.linear_1(x)
        output = self.relu(output)
        output = self.linear_2(output)
        return output

def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return Embeddings(config=config)
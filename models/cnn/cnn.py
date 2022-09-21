import sys
import os
import json
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, config):
        """CNN model with global average pooling

        Args:
            config (dict): parameters
        """
        super(CNN, self).__init__()

        self.config = config
        self.layers_conv = self.config["layers"]
        self.channels = self.config["channels"]
        self.kernel_size = self.config["kernel_size"]
        self.strides = self.config["strides"]
        self.padding = self.config["padding"]
        self.pool_size = self.config["pool_size"]
        self.drop_prob_conv = self.config["drop_prob"]

        self.features = self.create_network()

    def create_network(self):
        modules_conv = []
        in_channels = 1

        for i_conv in range(self.layers_conv):
            modules_conv.append(
                nn.Conv2d(in_channels, self.channels[i_conv],
                          kernel_size=self.kernel_size[i_conv],
                          stride=self.strides[i_conv],
                          padding=self.padding))
            modules_conv.append(nn.BatchNorm2d(self.channels[i_conv]))
            modules_conv.append(nn.ReLU())
            modules_conv.append(nn.MaxPool2d((self.pool_size[0],self.pool_size[1])))
            modules_conv.append(nn.Dropout(p=self.drop_prob_conv))
            in_channels = self.channels[i_conv]
        
        modules_conv.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        return nn.Sequential(*modules_conv)
    
    def forward(self, x):
        out = self.features(x)
        return out
    
def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return CNN(config=config)
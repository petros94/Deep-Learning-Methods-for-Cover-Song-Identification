import torch
import torchvision
from torch import nn

import json

class ResNet18(nn.Module):
  def __init__(self, out_channels, kernel_size, stride, padding) -> None:
    super(ResNet18, self).__init__()
    # get resnet model
    self.resnet = torchvision.models.resnet18(pretrained=False)
    # over-write the first conv layer to be able to read MNIST images
    # as resnet18 reads (3,x,x) where 3 is RGB channels
    # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
    self.resnet.conv1 = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.fc_in_features = self.resnet.fc.in_features
    
    # remove the last layer of resnet18 (linear layer which is before avgpool layer)
    self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    # initialize the weights
    self.resnet.apply(self.init_weights)
        
  def init_weights(self, m):
      if isinstance(m, nn.Linear):
          torch.nn.init.xavier_uniform(m.weight)
          m.bias.data.fill_(0.01)

  def forward_once(self, x):
      output = self.resnet(x)
      output = output.view(output.size()[0], -1)
      return output

  def forward(self, x):
      output1 = self.forward_once(x)
      return output1
  
def from_config(config_path: str):
    """Create a resnet18 model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return ResNet18(out_channels=config['out_channels'],
                        kernel_size=config['kernel_size'], 
                        stride=config['stride'], 
                        padding=config['padding'])
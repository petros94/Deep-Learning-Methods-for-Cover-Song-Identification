import torch
import torchvision
from torch import nn

import json

class EfficientNet(nn.Module):
  def __init__(self) -> None:
    super(EfficientNet, self).__init__()
    # get resnet model
    self.efficient_net = torchvision.models.efficientnet_b0()
    
    print(self.efficient_net)
    print(list(self.efficient_net.children()))
    # over-write the first conv layer to be able to read MNIST images
    # as resnet18 reads (3,x,x) where 3 is RGB channels
    # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
    # self.resnet.conv1 = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    # remove the last layer of resnet18 (linear layer which is before avgpool layer)
    # self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))


  def forward_once(self, x):
      output = self.efficient_net(x)
      output = output.view(output.size()[0], -1)
      return output

  def forward(self, x):
      output1 = self.forward_once(x)
      return output1
  
def from_config(config_path: str):
    """Create an efficient net model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return EfficientNet()
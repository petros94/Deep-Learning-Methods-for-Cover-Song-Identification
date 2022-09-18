from torch import nn
import torch

class ThresholdClassifier(nn.Module):
    def __init__(self, model, D) -> None:
        super(ThresholdClassifier, self).__init__()
        self.D = D
        self.model = model
        
    def forward(self, x1, x2):
        out1 = self.model(x1)
        out2 = self.model(x2)
        dist = torch.norm(out1 - out2, dim=1)
        return (dist < self.D)*1
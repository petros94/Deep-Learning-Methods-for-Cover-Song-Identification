import torch
import torch.nn as nn
import json
from transformers import Wav2Vec2Processor, HubertModel

from utils.generic import get_device

device = get_device()

class MERT(nn.Module):
    def __init__(self, config):
        super(MERT, self).__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        # loading our model weights
        self.model = HubertModel.from_pretrained("m-a-p/MERT-v0")

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(-1)
        x = x.view(batch_size, sequence_length)
        x = torch.FloatTensor(x).to(device)

        outputs = self.model(input_values=x, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)

        aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
        weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
        return weighted_avg_hidden_states

def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return MERT(config=config)
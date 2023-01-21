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
        for param in self.model.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=768, hidden_size=512, batch_first=True)

        self.cache = {}

    def forward(self, x, song_ids=None):
        batch_size = x.size(0)
        sequence_length = x.size(-1)
        x = x.view(batch_size, sequence_length)
        x = torch.tensor(x, dtype=torch.float32).to(device)

        hash_key = None
        if song_ids is not None:
            hash_key = str(song_ids)

        if hash_key is not None and hash_key in self.cache.keys():
            time_reduced_hidden_states = self.cache[hash_key].to(device)
        else:
            with torch.no_grad():
                outputs = self.model(input_values=x, output_hidden_states=True)
                all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
                time_reduced_hidden_states = all_layer_hidden_states.mean(0).squeeze(0)
            print("added to cache")
            self.cache[hash_key] = time_reduced_hidden_states.to("cpu")

        output, (h_n, c_n) = self.lstm(time_reduced_hidden_states)
        output = output[:, -1, :]
        return output

def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return MERT(config=config)
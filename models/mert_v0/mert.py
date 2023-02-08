import torch
import torch.nn as nn
import json
from transformers import Wav2Vec2Processor, HubertForSequenceClassification, HubertConfig

from utils.generic import get_device

device = get_device()

class MERT(nn.Module):
    def __init__(self, config):
        super(MERT, self).__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        # loading our model weights
        self.config = HubertConfig.from_pretrained("m-a-p/MERT-v0")
        self.config.num_labels = 128
        self.model = HubertForSequenceClassification.from_pretrained("m-a-p/MERT-v0", config=self.config)
        self.model.freeze_feature_encoder()
        self.model.freeze_base_model()

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(-1)
        x = x.view(batch_size, sequence_length)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        outputs = self.model(input_values=x, output_hidden_states=True).logits
        return outputs

def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return MERT(config=config)
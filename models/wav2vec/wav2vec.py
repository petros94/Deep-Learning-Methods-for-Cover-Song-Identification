import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd

from utils.generic import get_device

device = get_device()

class Wav2Vec2(nn.Module):
    def __init__(self, config):
        super(Wav2Vec2, self).__init__()
        model_name_or_path = "m3hrdadfi/wav2vec2-base-100k-voxpopuli-gtzan-music"
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        self.pretrained_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name_or_path).to(device)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(self.model_config.hidden_size, config["hidden_size"])

    def forward(self, x):
        inputs = self.feature_extractor(x, sampling_rate=22050, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(device) for key in inputs}

        with torch.no_grad():
            last_hidden_layer_emb = self.pretrained_model(**inputs).last_hidden_state
            last_conv_layer_emb = self.pretrained_model(**inputs).extract_features

        output = self.linear(last_hidden_layer_emb)
        return output

def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return Wav2Vec2(config=config)
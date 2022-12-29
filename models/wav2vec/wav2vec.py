import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import torchaudio
import time
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
        model_name_or_path = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        self.pretrained_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name_or_path, output_hidden_states=True).to(device)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(self.model_config.hidden_size, config["hidden_size"]).to(device)

    def forward(self, x):
        x = x.squeeze(1).squeeze(1)
        values = []
        for it in x:
            inputs = self.feature_extractor(it, sampling_rate=16000, return_tensors="pt", padding=False)
            values.append(inputs['input_values'])
        inputs = {'input_values': torch.FloatTensor(torch.cat(values, dim=0)).to(device)}


        with torch.no_grad():
            now = time.time()
            print("Now running wav2vec inference")
            wav2vec_output = self.pretrained_model(**inputs)
            print(f"Finished wav2vec inference in {(time.time() - now)} seconds")
            last_hidden_layer_emb = torch.mean(wav2vec_output.hidden_states[-1], dim=1)
            print(last_hidden_layer_emb.size())
            # last_conv_layer_emb = self.pretrained_model(**inputs).extract_features

        output = self.linear(last_hidden_layer_emb)
        print(output.size())
        return output

def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return Wav2Vec2(config=config)
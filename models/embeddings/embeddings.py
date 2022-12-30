import torch
import torch.nn as nn
import json
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from utils.generic import get_device

device = get_device()

class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.linear_1 = nn.Linear(config["input_size"], config["hidden_size"]).to(device)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(config["hidden_size"], config["hidden_size"]).to(device)

    def calculate_embeddings(self, dataset):
        device = get_device()
        model_name_or_path = "m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres"
        wav2vec_model_config = AutoConfig.from_pretrained(model_name_or_path)
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        wav2vec_pretrained_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name_or_path,
                                                                                     output_hidden_states=True).to(
            device)

        for param in wav2vec_pretrained_model.parameters():
            param.requires_grad = False

        embeddings = []
        labels = []
        song_names = []
        for i in range(len(dataset)):
            frames, label, song_name = dataset[i]
            frames = torch.tensor(frames).squeeze(0).squeeze(0).squeeze(0).to(device)
            print(frames.size())

            inputs = wav2vec_feature_extractor(frames, sampling_rate=16000, return_tensors="pt", padding=False)
            inputs['input_values'] = torch.FloatTensor(inputs['input_values']).to(device)
            with torch.no_grad():
                wav2vec_output = wav2vec_pretrained_model(**inputs)
                last_hidden_layer_emb = torch.mean(wav2vec_output.hidden_states[-1], dim=1)
                # last_conv_layer_emb = self.pretrained_model(**inputs).extract_features
                embeddings.append(last_hidden_layer_emb)
                labels.append(label)
                song_names.append(song_name)

        return torch.stack(embeddings, dim=0).squeeze(1), torch.tensor(labels), song_names
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
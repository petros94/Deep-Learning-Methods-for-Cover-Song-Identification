from torch import nn
import torch
import json
from torch.nn import functional as F


class LSTMCNN(nn.Module):
    def __init__(
        self, config, bidirectional=False
    ) -> None:
        super(LSTMCNN, self).__init__()
        self.config = config
        self.in_channels = self.config["in_channels"]
        self.layers_conv = self.config["layers"]
        self.channels = self.config["channels"]
        self.kernel_size = self.config["kernel_size"]
        self.strides = self.config["strides"]
        self.padding = self.config["padding"]
        self.pool_size = self.config["pool_size"]
        self.drop_prob_conv = self.config["drop_prob"]
        self.input_size = self.config["input_size"]
        self.num_layers = self.config["num_layers"]
        self.hidden_size = self.config["hidden_size"]
        self.bidirectional = self.config["bidirectional"]
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            dropout=0.2,
        )
        self.output_size = self.config['output_size']
        self.fc = nn.Linear(
            in_features=(2 if bidirectional else 1) * self.hidden_size,
            out_features=self.output_size,
        )
        self.dropout = nn.Dropout(0.2)

        self.embeddings = {}
        self.embedding_layer = self.create_cnn_layers()

    def create_cnn_layers(self):
        modules_conv = []
        in_channels = self.in_channels

        for i_conv in range(self.layers_conv):
            modules_conv.append(
                nn.Conv1d(in_channels, self.channels[i_conv],
                          kernel_size=self.kernel_size[i_conv],
                          stride=self.strides[i_conv],
                          padding=self.padding))
            modules_conv.append(nn.BatchNorm1d(self.channels[i_conv]))
            modules_conv.append(nn.ReLU())
            modules_conv.append(nn.MaxPool1d(self.pool_size))
            modules_conv.append(nn.Dropout(p=self.drop_prob_conv))
            in_channels = self.channels[i_conv]

        return nn.Sequential(*modules_conv)
    def forward(self, x):
        """Forwars

        Args:
            x (tensor): B X 1 X num_feats X num_samples

        Returns:
            tensor: N
        """
        x = x.squeeze(1)
        emb = self.embedding_layer(x)
        emb = emb.permute(2, 0, 1)

        out_packed, (h, c) = self.lstm(emb)
        out_packed = out_packed[-1, :, :]
        out = self.fc(out_packed)
        return out


def from_config(config_path: str):
    """Create a cnn model from configuration

    Args:
        config_path (str): The path to the configuration file
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        return LSTMCNN(config)

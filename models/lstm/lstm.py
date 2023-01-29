from torch import nn
import torch
import json
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, bidirectional=False
    ) -> None:
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            dropout=0.2,
        )
        self.output_size = output_size
        self.fc = nn.Linear(
            in_features=(2 if bidirectional else 1) * self.hidden_size,
            out_features=output_size,
        )
        self.dropout = nn.Dropout(0.2)

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, x):
        """Forwars

        Args:
            x (tensor): B X 1 X num_feats X num_samples

        Returns:
            tensor: N
        """
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        out_packed, (h, c) = self.lstm(x)
        # attn_output = self.attention(out_packed, h)
        # cat = (
        #     torch.cat((h[-1, :, :], h[-2, :, :]), dim=1)
        #     if self.bidirectional
        #     else h[-1, :, :]
        # )
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
        return LSTM(**config)

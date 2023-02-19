import json
import logging
import time
import timeit
import traceback

import torch

from execute_experiment import evaluate_model
from models.cnn.cnn import from_config as make_audio_model
from songbase.load_songs import from_config
from utils.generic import get_device

if __name__ == '__main__':
    config_path = "experiments/evaluation_pretrained.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Loading songs")
    test_songs = from_config(config_path=config_path)
    reprs = [torch.tensor(cover['repr'])[:, :, :200] for song in test_songs.keys() for cover in test_songs[song]]
    reprs = torch.stack(reprs).to(get_device())
    print(reprs.size())
    audio_model = make_audio_model(config["model"]["config_path"])
    if config["model"]["checkpoint_path"] is not None:
        loc = get_device()
        chk = torch.load(
            config["model"]["checkpoint_path"], map_location=torch.device(loc)
        )
        print("loaded pretrained model")

        audio_model.load_state_dict(chk["model_state_dict"])
        audio_model.eval()
        audio_model.to(get_device())

    for i in range(10):
        with torch.no_grad():
            st = time.time()
            out = audio_model(reprs)
            et = time.time()

        print(f"Total time {et-st} seconds")
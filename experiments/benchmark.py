import time

import numpy as np
import pandas as pd
import torch

from datasets.TripletDataset import TripletDataset
from models.factory import make_model
from songbase.load_songs import from_config
from utils.generic import get_device


from datetime import datetime
import json
import os

def benchmark_device(config_path: str = "experiments/evaluation_pretrained.json", device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Creating directory for checkpoint and saving configuration used")
    date_ = f"{datetime.now()}_{config['model']['type']}_{config['features']['input']}_{config['loss']}"
    res_dir_name = config["train"]["results_path"] + date_
    os.mkdir(res_dir_name)

    print("Evaluating for segmented input")
    res_dir_name = res_dir_name + '/segmented'

    os.mkdir(res_dir_name)

    with open(config_path, "r") as f:
        config = json.load(f)

    model = make_model(config_path=config_path).to(device)
    model.eval()

    print("Loading songs")
    _, test_songs = from_config(config_path=config_path)

    frame_sizes = []
    durations = []
    total_durations = []

    print("warmup")
    test_set = TripletDataset(test_songs,
                              n_batches=512,
                              songs_per_batch=16,
                              frame_size=12000,
                              scale=[1, 0.2])
    for i in range(len(test_set)):
        (data, labels) = test_set[i]
        data = data.type(torch.FloatTensor).to(device)
        data = data[0].unsqueeze(0)
        embeddings = model(data)


    print("actual test")
    for frame_size in (750, 1500, 3000, 6000, 12000):
        test_set = TripletDataset(test_songs,
                                    n_batches=512,
                                    songs_per_batch=16,
                                    frame_size=frame_size,
                                    scale=[1, 0.2])
        count = 0
        frame_sizes.append(frame_size)
        samples = []
        print("Inference")
        for i in range(len(test_set)):
            count += 1
            (data, labels) = test_set[i]
            data = data.type(torch.FloatTensor).to(device)
            data = data[0].unsqueeze(0)
            before = time.time()
            embeddings = model(data)
            duration = time.time() - before
            samples.append(duration)
        print("Done")

        durations.append(np.mean(samples))
        total_durations.append(np.sum(samples))

    df = pd.DataFrame({'frame_size': frame_sizes,
                       'mean_duration': durations,
                       'total_duration': total_durations,
                       'count': len(frame_sizes)*[count],
                       'device': len(frame_sizes)*[device]})
    return df

def benchmark():
    print("CPU benchmark")
    df_cpu = benchmark_device(device="cpu")

    print("GPU benchmark")
    df_gpu = benchmark_device(device=get_device())

    df = pd.concat([df_cpu, df_gpu])
    print(df)



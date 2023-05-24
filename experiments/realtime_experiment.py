import time
from re import L

from datasets.TripletDataset import TripletDataset
from datasets.factory import make_dataset
from feature_extraction.downloader import YoutubeDownloader
from feature_extraction.extraction import FeatureExtractor
from models.classifier import ThresholdClassifier
from models.factory import make_model
import os
import uuid
import json
import torch
import shutil
from datetime import datetime

from songbase.load_songs import from_config
from utils.generic import get_device, segment_and_scale


def download_songs_and_inference(config_path: str, links: list, tmp_base_dir="/content"):
    device = get_device()
    with open(config_path, "r") as f:
        config = json.load(f)

    model = make_model(config_path=config_path)
    thr = config['model']['threshold']
    clf = ThresholdClassifier(model, thr)
    clf.eval()
    clf.to(device)

    scale = (1, 0.1)
    frame_size = 4400
    idx = 0

    # Download songs to temp dir
    downloader = YoutubeDownloader()
    temp_dir = tmp_base_dir + "/" + str(uuid.uuid4())
    os.makedirs(temp_dir)
    for link in links:
        downloader.download(path=temp_dir, link=link)

    # Extract features
    extractor = FeatureExtractor(features=config['representation'][0])
    entries = os.listdir(temp_dir)
    features = {}
    for song in entries:
        feat = extractor.extract(os.path.realpath(song))

        # tensor of size num_segs X num_channels X num_features X num_samples
        features[song] = segment_and_scale(feat, frame_size, scale)

    # Forward pass
    for song_1, value_1 in features.items():
        for song_2, value_2 in features.items():
            if song_1 != song_2:
                x_1 = torch.tensor(value_1[idx]).unsqueeze(0).to(device)
                x_2 = torch.tensor(value_2[idx]).unsqueeze(0).to(device)

                pred, dist = clf(x_1, x_2)
                print(
                    f"- Song 1: {song_1}, Song 2: {song_2} \n are covers: {pred.item() == True}, distance: {dist.item()}")
        print("")
    # Remove temp dir
    shutil.rmtree(temp_dir)


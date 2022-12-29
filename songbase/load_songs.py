import os
import random
import time
import json

import numpy as np
import scipy.io
from torch.nn import functional as F
from utils.generic import sample_songs


def from_config(config_path: str = "songbase/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    train_songs = {}
    try:
        for dataset in config["train_datasets"]:
            songs = load_songs(
                type=dataset["type"],
                songs_dir=dataset["path"],
                features=config["representation"],
            )
            songs = sample_songs(songs, n_samples=dataset["n_samples"])
            train_songs.update(songs)
    except KeyError:
        print("No train datasets supplied.")

    test_songs = {}
    try:
        for dataset in config["test_datasets"]:
            songs = load_songs(
                type=dataset["type"],
                songs_dir=dataset["path"],
                features=config["representation"],
            )
            songs = sample_songs(songs, n_samples=dataset["n_samples"])
            test_songs.update(songs)
    except KeyError:
        print("No test datasets supplied.")

    return train_songs, test_songs


def load_songs(type="covers1000", songs_dir=["mfccs/"], features=["mfcc"]):
    """
    Load the song database in JSON format.

    Example:
    {
        "120345 (song_id)": [
            {
                "song_id": "120345",
                "cover_id": "454444",
                "repr": [[3.43, 2.34, 5.55, ...], [4.22, 0.45, 3.44], ...]  #num_features X num_samples
            },
            ...
        ]
    }

    Arguments:
        type: ["covers1000", "covers80"] the dataset type
    """
    if type == "covers1000":
        return load_songs_covers1000(songs_dir, features)
    elif type == "covers80":
        return load_songs_covers80(songs_dir, features)
    else:
        raise ValueError("'type' must be one of ['covers1000', 'covers80']")


def load_songs_covers1000(songs_dir=["mfccs/"], features=["mfcc"]):
    songs = {}
    for song_dir, feature in zip(songs_dir, features):
        songs[feature] = {}
        origin_path = song_dir
        entries = os.listdir(origin_path)

        if feature == "mfcc":
            mat_feature = "XMFCC"
        elif feature == "hpcp":
            mat_feature = "XHPCP"
        elif feature == "cens":
            mat_feature = "XCENS"

        for dir in entries:
            subdir = os.listdir(origin_path + dir)
            songs[feature][dir] = []
            for song in subdir:
                song_id = dir
                cover_id = song.split("_")[0]
                mat = scipy.io.loadmat(origin_path + dir + "/" + song)
                repr = mat[mat_feature]
                repr = np.array(repr)
                repr = (repr - np.mean(repr)) / np.std(repr)
                songs[feature][dir].append(
                    {"song_id": song_id, "cover_id": cover_id, "repr": repr}
                )

    return merge_song_representations(songs)


def load_songs_covers80(songs_dir=["hpcps80/"], features=["hpcp"]):
    songs = {}
    skipped_counter = 0
    for song_dir, feature in zip(songs_dir, features):
        songs[feature] = {}
        origin_path = song_dir
        entries = os.listdir(origin_path)

        if feature == "mfcc":
            mat_feature = "XMFCC"
        elif feature == "hpcp":
            mat_feature = "XHPCP"
        elif feature == "cens":
            mat_feature = "XCENS"
        elif feature == "wav":
            mat_feature = "XWAV"

        for dir in entries:
            subdir = os.listdir(origin_path + dir)
            
            if len(subdir) <= 1:
                skipped_counter += 1
                print(
                    f"Warning found song with no covers: {origin_path + dir}, skipping..."
                )
                continue
            
            songs[feature][dir] = []

            for song in subdir:
                song_id = dir
                cover_id = song
                mat = scipy.io.loadmat(origin_path + dir + "/" + song)
                repr = mat[mat_feature]  # No need to normalize since already normalized
                repr = np.array(repr)
                songs[feature][dir].append(
                    {"song_id": song_id, "cover_id": cover_id, "repr": repr}
                )

    print(f"Total: {len(songs[features[0]])}, skipped: {skipped_counter}")
    return merge_song_representations(songs)

def merge_song_representations(songs):
    """Merge song representations.

    Args:
        songs (dict): songs in the following format:
        {
            'mfcc': {
                'song_id': [
                    {"song_id": 'song_id', "cover_id": 'cover_id_1', "repr": repr_11},
                    {"song_id": 'song_id', "cover_id": 'cover_id_2', "repr": repr_12},
                    {"song_id": 'song_id', "cover_id": 'cover_id_3', "repr": repr_13}
                ],
                'song_id_2': [...]
                ...
            },
            'hpcp': {
                'song_id': [
                    {"song_id": 'song_id', "cover_id": 'cover_id_1', "repr": repr_21},
                    {"song_id": 'song_id', "cover_id": 'cover_id_2', "repr": repr_22},
                    {"song_id": 'song_id', "cover_id": 'cover_id_3', "repr": repr_23}
                ],
                'song_id_2': [...],
                ...
            }
        }

    Returns:
        concatenated song dict:
        {
            'song_id': [
                    {"song_id": 'song_id', "cover_id": 'cover_id_1', "repr": [repr_11, repr_21]},
                    {"song_id": 'song_id', "cover_id": 'cover_id_2', "repr": [repr_12, repr_22]},
                    {"song_id": 'song_id', "cover_id": 'cover_id_3', "repr": [repr_13, repr_23]}
            ],
            'song_id_2': [...]
            ...
        }
    """
    features = list(songs.keys())
    anchor_feat = features[0]
    songs_ids = list(songs[anchor_feat].keys())
    songs_concatenated_features = {}
    for id in songs_ids:
        songs_concatenated_features[id] = []

        covers = [songs[feature][id] for feature in features]
        for feats in zip(*covers):
            song_id = feats[0]["song_id"]
            cover_id = feats[0]["cover_id"]
            repr = [r["repr"] for r in feats]
            songs_concatenated_features[id].append(
                {"song_id": song_id, "cover_id": cover_id, "repr": repr}
            )
    return songs_concatenated_features

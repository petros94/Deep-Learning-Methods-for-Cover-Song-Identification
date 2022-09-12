import os 
import random
import time
import json

import numpy as np
import scipy.io
from torch.nn import functional as F
from utils.generic import sample_songs

def from_config(config_path: str = 'songbase/config.json'):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    ret = {}
    for dataset in config['datasets']:
        songs = load_songs(type=dataset['type'], 
                            songs_dir=dataset['path'], 
                            feature=config['representation'])
        songs = sample_songs(songs, n_samples=dataset['n_samples'])
        ret.update(songs)
        
    return ret
         

def load_songs(type="covers1000", songs_dir="mfccs/", feature="mfcc"):
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
        type: ["covers1000"] the dataset type
    """
    if type == "covers1000":
        return load_songs_covers1000(songs_dir, feature)
    else:
        raise ValueError("'type' must be one of ['covers1000']")

def load_songs_covers1000(songs_dir="mfccs/", feature="mfcc"):
    origin_path = songs_dir
    entries = os.listdir(origin_path)
    
    songs = {}
    if feature == "mfcc":
        feature = 'XMFCC'

    for dir in entries:
        subdir = os.listdir(origin_path + dir)
        songs[dir] = []
        for song in subdir:
            song_id = dir
            cover_id = song.split("_")[0]
            mat = scipy.io.loadmat(origin_path + dir + "/" + song)
            repr = mat[feature]
            repr = np.array(repr)
            repr = (repr - np.mean(repr)) / np.std(repr)
            songs[dir].append({"song_id": song_id, "cover_id": cover_id, "repr": repr})
    return songs

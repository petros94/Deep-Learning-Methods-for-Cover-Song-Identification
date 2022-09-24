from dataclasses import make_dataclass

import json

from datasets.RandomTripletDataset import RandomTripletDataset
from datasets.HardTripletDataset import HardTripletDataset

AVAILABLE_TYPES = ["triplet", "hard_triplet"]

def make_dataset(songs: dict, config_path: str, type: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if type not in AVAILABLE_TYPES:
        raise ValueError("Invalid type")
    
    if type == 'triplet':
        return RandomTripletDataset(songs, 
                              samples_per_song=config['features']['samples_per_song'], 
                              frame_size=config['features']['frame_size'],
                              scale=config['features']['scale'])
    elif type == 'hard_triplet':
        return HardTripletDataset(songs, 
                                  n_batches=config['features']['n_batches'], 
                                  songs_per_batch=config['features']['songs_per_batch'],
                                  frame_size=config['features']['frame_size'],
                                  scale=config['features']['scale'])
        
        
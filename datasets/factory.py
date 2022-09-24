from dataclasses import make_dataclass

import json

from datasets.RandomTripletDataset import RandomTripletDataset
from datasets.HardTripletDataset import HardTripletDataset

AVAILABLE_TYPES = ["triplet", "hard_triplet"]

def make_dataset(songs: dict, config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if config['loss'] not in AVAILABLE_TYPES:
        raise ValueError("Invalid type")
    
    if config['loss'] == 'triplet':
        return RandomTripletDataset(songs, 
                              samples_per_song=config['features']['samples_per_song'], 
                              frame_size=config['features']['frame_size'],
                              scale=config['features']['scale'])
    elif config['loss'] == 'hard_triplet':
        return HardTripletDataset(songs, 
                                  n_batches=config['features']['n_batches'], 
                                  songs_per_batch=config['features']['songs_per_batch'],
                                  frame_size=config['features']['frame_size'],
                                  scale=config['features']['scale'])
        
        
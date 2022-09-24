from dataclasses import make_dataclass

import json

from datasets.RandomTripletDataset import RandomTripletDataset

AVAILABLE_TYPES = ["triplet"]

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
        
        
import json

from datasets.TripletDataset import TripletDataset

AVAILABLE_TYPES = ["triplet"]

def make_dataset(songs: dict, config_path: str, type: str):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if type not in AVAILABLE_TYPES:
        raise ValueError("Invalid type")
    
    if type == 'triplet':
        return TripletDataset(songs, 
                              samples_per_song=config['features']['samples_per_song'], 
                              frame_size=config['features']['frame_size'],
                              scale=config['features']['scale'])       
        
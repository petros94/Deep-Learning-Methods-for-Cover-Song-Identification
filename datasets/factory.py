import json
from datasets.SimpleDataset import SimpleDataset

from datasets.TripletDataset import TripletDataset

AVAILABLE_TYPES = ["triplet", "angular"]

def make_dataset(songs: dict, config_path: str, type: str, segmented: bool, n_batches=None):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if type not in AVAILABLE_TYPES:
        raise ValueError("Invalid type")

    if config['model']['type'] == "embeddings":
        return SimpleDataset(songs, config['features']['scale'])

    if segmented:
        if type in ('triplet', 'angular'):
            return [TripletDataset(songs,
                                n_batches=n_batches if n_batches is not None else config['features']['n_batches'], 
                                songs_per_batch=config['features']['songs_per_batch'],
                                frame_size=frame_size,
                                scale=config['features']['scale'],
                                no_augment=config['features']['no_augment']) for frame_size in config['features']['frame_size']]
    else:
        return SimpleDataset(songs, config['features']['scale'])
        
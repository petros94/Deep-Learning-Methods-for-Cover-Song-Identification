import json
import logging
from datasets.factory import make_dataset

from models.factory import make_model
from songbase.load_songs import from_config
from training.train import train
from utils.generic import split_songs

def execute_single(config_path: str = 'experiments/experiment_config.json'):
    logging.info("Executing single experiment")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    logging.info("Loading songs")
    songs = from_config(config_path=config_path)
    
    train_songs, valid_songs = split_songs(songs, config['train']['train_perc'])
    
    train_set = make_dataset(train_songs, config_path=config_path)
    valid_set = make_dataset(valid_songs, config_path=config_path)
    logging.info("Created training set: {} samples, valid set: {} samples", len(train_set), len(valid_set))
    
    model = make_model(config_path=config_path)
    
    logging.info("Begin training")
    train(model, train_set, valid_set, config_path=config_path)   

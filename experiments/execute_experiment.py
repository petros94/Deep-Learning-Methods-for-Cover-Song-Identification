import json
import os
import logging
from datasets.factory import make_dataset
from datetime import datetime

from models.factory import make_model
from songbase.load_songs import from_config
from training.train import train
from utils.generic import split_songs

def execute_single(config_path: str = 'experiments/experiment_config.json'):
    print("Executing single experiment")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print("Creating directory for checkpoint and saving configuration used")
    dir_name = config['train']['checkpoints_path'] + f"{datetime.now()}_{config['model']['type']}_{config['features']['input']}_{config['loss']}"
    os.mkdir(dir_name)
    with open(dir_name + "/config.json", "w") as f:
        json.dump(config, f)
        
    print("Loading songs")
    songs = from_config(config_path=config_path)
    
    train_songs, valid_songs = split_songs(songs, config['train']['train_perc'])
    
    train_set = make_dataset(train_songs, config_path=config_path)
    valid_set = make_dataset(valid_songs, config_path=config_path)
    print("Created training set: {} samples, valid set: {} samples".format(len(train_set), len(valid_set)))
    
    model = make_model(config_path=config_path)
    
    print("Begin training")
    train(model, train_set, valid_set, config_path=config_path, checkpoint_dir=dir_name)   

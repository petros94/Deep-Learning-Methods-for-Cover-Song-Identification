import json
import os
import logging
import random
import numpy as np
from datasets.factory import make_dataset
from datetime import datetime

from models.factory import make_model
from models.threshold import Threshold
from songbase.load_songs import from_config
from training.train import train
from utils.generic import get_device, split_songs
from utils.prediction import distance
import torch

def execute_single(config_path: str = 'experiments/experiment_config.json'):
    print("Executing single experiment")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print("Creating directory for checkpoint and saving configuration used")
    date_ = f"{datetime.now()}_{config['model']['type']}_{config['features']['input']}_{config['loss']}"
    chk_dir_name = config['train']['checkpoints_path'] + date_
    os.mkdir(chk_dir_name)
    with open(chk_dir_name + "/config.json", "w") as f:
        json.dump(config, f)
        
    res_dir_name = config['train']['results_path'] + date_
    os.mkdir(res_dir_name)
        
    print("Loading songs")
    songs = from_config(config_path=config_path)
    
    train_songs, valid_songs = split_songs(songs, config['train']['train_perc'])
    
    train_set = make_dataset(train_songs, config_path=config_path)
    valid_set = make_dataset(valid_songs, config_path=config_path)
    print("Created training set: {} samples, valid set: {} samples".format(len(train_set), len(valid_set)))
    
    model = make_model(config_path=config_path)
    
    print("Begin training")
    train(model, train_set, valid_set, config_path=config_path, checkpoint_dir=chk_dir_name, results_dir=res_dir_name) 
    
    print("Begin validation set testing")
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=getattr(valid_set, "collate_fn", None))
    model.eval()
    device = get_device()
    model.to(device)
    output = torch.Tensor()
    labels = []
    with torch.no_grad():
        for batch, (x, metadata) in enumerate(valid_dataloader):     
        
            # x: 3 x N x 1 x W x H
            (anchor, pos, neg) = x 

            anchor.to(device)
            pos.to(device)
            neg.to(device)
            
            pair, label = (anchor, pos), 1 if random.random() > 0.5 else (anchor, neg), 0
            
            #first dimension: N X 128
            first, second = model(pair[0]), model(pair[1])
            
            distances = torch.norm(first - second)
            output = torch.cat((output, distances))
            labels.extend(label)
            
    output, labels = output.cpu().numpy(), np.array(labels)
    print(len(output))
    
    tr = Threshold(0.5)
    tr.generate_ROC(output, labels)
                


      

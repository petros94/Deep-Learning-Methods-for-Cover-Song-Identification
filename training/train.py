import json

from training.train_cnn_triplet import train_triplet_loss as cnn_triplet
from training.train_lstm_triplet import train_triplet_loss as lstm_triplet

def train(model, train_set, valid_set, config_path, checkpoint_dir, results_dir):
    with open(config_path, "r") as f:
        config = json.load(f)
                
    if config['loss'] == 'triplet':
        if config['model']['type'] == 'cnn':
            return cnn_triplet(model, train_set, valid_set, 
                                    config['train']['n_epochs'], config['train']['patience'], config['train']['lr'], 
                                    checkpoint_dir, results_dir)
        elif config['model']['type'] == 'lstm':
            return lstm_triplet(model, train_set, valid_set, 
                                    config['train']['n_epochs'], config['train']['patience'], config['train']['lr'], 
                                    checkpoint_dir, results_dir)
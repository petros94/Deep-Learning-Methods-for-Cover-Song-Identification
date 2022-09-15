import json

from training.train_cnn_triplet import train_triplet_loss

def train(model, train_set, valid_set, config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if config['loss'] == 'triplet' and config['model']['type'] in ('resnet18'):
        return train_triplet_loss(model, train_set, valid_set, config['train']['n_epochs'], config['train']['batch_size'], config['train']['lr'], config['device'])
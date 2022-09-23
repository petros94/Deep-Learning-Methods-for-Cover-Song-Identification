import json
from models.resnet18.ResNet18 import from_config as make_resnet
from models.cnn.cnn import from_config as make_cnn
import torch
from utils.generic import get_device


AVAILABLE_TYPES = ["resnet18", "cnn"]

def make_model(config_path: str = 'models/config.json'):
    """Factory to generate models from configuration

    Args:
        type (str): the model type. Check available_types above
        config_path (str): the configuration path

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if config['model']['type'] not in AVAILABLE_TYPES:
        raise ValueError("Invalid type")
    
    if config['model']['type'] == "resnet18":
        model = make_resnet(config_path=config['model']['config_path'])
    elif config['model']['type'] == "cnn":
        model = make_cnn(config_path=config['model']['config_path'])
        
    if config['model']['checkpoint_path'] is not None:
        loc = get_device()
        chk = torch.load(config['model']['checkpoint_path'], map_location=torch.device(loc)) 
        
        model.load_state_dict(chk['model_state_dict'])
        epoch = chk['epoch']
        loss = chk['loss']
        model.eval()
        
    return model
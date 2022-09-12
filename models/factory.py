import json
from models.resnet18.ResNet18 import from_config as make_resnet
import json

AVAILABLE_TYPES = ["resnet18"]

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
    
    if config['model']['type'] == "resnet":
        return make_resnet(config_path=config['model']['config_path'])
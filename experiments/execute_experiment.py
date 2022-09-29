import json
import os
import numpy as np
from datasets.factory import make_dataset
from datetime import datetime
from models.classifier import ThresholdClassifier

from models.factory import make_model
from songbase.load_songs import from_config
from training.train import train
from utils.generic import split_songs
from utils.visualization import visualize_losses
from tuning.tuning import generate_ROC, generate_metrics

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
    train_songs, test_songs = from_config(config_path=config_path)
    
    train_songs, valid_songs = split_songs(train_songs, config['train']['train_perc'])
    
    train_set = make_dataset(train_songs, config_path=config_path, type="hard_triplet")
    valid_set = make_dataset(valid_songs, config_path=config_path, type="hard_triplet")

    if len(test_songs) > 0:
        test_set = make_dataset(test_songs, config_path=config_path, type="hard_triplet")
    else:
        print("No test set provided, validation set will be used")
        test_set = valid_set
        
    print("Created training set: {} samples, valid set: {} samples".format(len(train_set), len(valid_set)))
    
    model = make_model(config_path=config_path)
    
    print("Begin training")
    losses = train(model, train_set, valid_set, config_path=config_path, checkpoint_dir=chk_dir_name, results_dir=res_dir_name) 
        
    print("Plot losses")
    visualize_losses(losses, file_path=res_dir_name)
    
    print("Plot ROC and calculate metrics")
    roc_stats = generate_ROC(model, test_set, config['train']['batch_size'], results_path=res_dir_name)
    
    try:
        thr = config['model']['threshold']
    except KeyError:
        thr = roc_stats.loc[roc_stats['tpr'] > 0.7, 'thr'].iloc[0]
    
    clf = ThresholdClassifier(model, thr)
        
    df = generate_metrics(clf, test_set, config['train']['batch_size'], results_path=res_dir_name)
    
    with open('results/template.html') as f:
        report_html = f.read()
        report_html = report_html.replace('__DATA__', json.dumps(config))
        
        with open(config['model']['config_path']) as f2:
            model_config = json.load(f2)
        report_html = report_html.replace('__MODELDATA__', json.dumps(model_config))
        report_html = report_html.replace('__TABLE__', df.to_html())
        
    with open(res_dir_name + '/report.html', 'w') as f:
        f.write(report_html)
        
    return res_dir_name, chk_dir_name

def evaluate_model(config_path: str = 'experiments/experiment_config.json'):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print("Creating directory for checkpoint and saving configuration used")
    date_ = f"{datetime.now()}_{config['model']['type']}_{config['features']['input']}_{config['loss']}"        
    res_dir_name = config['train']['results_path'] + date_
    os.mkdir(res_dir_name)
        
    print("Loading songs")
    _, test_songs = from_config(config_path=config_path)
    test_songs = make_dataset(test_songs, config_path=config_path, type='triplet')
    print("Created eval set: {} samples".format(len(test_songs)))
    
    model = make_model(config_path=config_path)
    
    print("Plot ROC and calculate metrics")
    roc_stats = generate_ROC(model, test_songs, config['train']['batch_size'], results_path=res_dir_name)
    
    try:
        thr = config['model']['threshold']
    except KeyError:
        thr = roc_stats.loc[roc_stats['tpr'] > 0.8, 'thr'].iloc[0]
    
    clf = ThresholdClassifier(model, thr)
    df = generate_metrics(clf, test_songs, config['train']['batch_size'], results_path=res_dir_name)
    
    with open('results/template.html') as f:
        report_html = f.read()
        report_html = report_html.replace('__DATA__', json.dumps(config))
        report_html = report_html.replace('__TABLE__', df.to_html())
        
    with open(res_dir_name + '/report.html', 'w') as f:
        f.write(report_html)
        
    return res_dir_name
    
    
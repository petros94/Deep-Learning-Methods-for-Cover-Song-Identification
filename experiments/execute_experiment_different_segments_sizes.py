import json
import os

import numpy as np
import pandas as pd

from datasets.factory import make_dataset
from datetime import datetime
from models.classifier import ThresholdClassifier

from models.factory import make_model
from songbase.load_songs import from_config
from training.train import train
from utils.generic import split_songs
from utils.visualization import visualize_losses
from tuning.ranking_metrics import generate_metrics as generate_ranking_metrics, \
    generate_ROC
from tuning.classification_metrics import generate_metrics as generate_classification_metrics

def evaluate_model(config_path: str = "experiments/evaluation_pretrained.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Loading songs")
    _, test_songs, _ = from_config(config_path=config_path)

    print("Creating directory for checkpoint and saving configuration used")
    date_ = f"{datetime.now()}_{config['model']['type']}_{config['features']['input']}_{config['loss']}"

    print("Evaluating balanced")
    res_dir_name = config["train"]["results_path"] + date_ + "_balanced"
    os.mkdir(res_dir_name)
    evaluate_test_set(config_path=config_path, results_path=res_dir_name, test_songs=test_songs)

    return res_dir_name


def evaluate_test_set(config_path, results_path, test_songs, model=None, valid_songs=None):
    res_dir_name = results_path + '/full'
    os.mkdir(res_dir_name)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if len(test_songs) > 0:
        test_set = make_dataset(test_songs, config_path=config_path, type=config["loss"], segmented=True, n_batches=128)
    else:
        print("No test set provided, validation set will be used")
        test_set = make_dataset(valid_songs, config_path=config_path, type=config["loss"], segmented=True, n_batches=128)

    if model is None:
        model = make_model(config_path=config_path)

    for frame_size, set in zip(config['features']['frame_size'] ,test_set):
        print(f"---------------Plot ROC and calculate metrics {frame_size}--------------------")
        roc_stats, _, _, _, _ = generate_ranking_metrics(model, set, True, results_path=res_dir_name, balanced=True)

        print(f"Scores: {roc_stats}")
        try:
            thr = config["model"]["threshold"]
        except KeyError:
            thr = roc_stats.loc[roc_stats["tpr"] > 0.8, "thr"].iloc[0]

        clf = ThresholdClassifier(model, thr)

        if config["representation"] == ['wav']:
            df = pd.DataFrame()
        else:
            df = generate_classification_metrics(
                clf, set, segmented=True, results_path=res_dir_name, balanced=True
            )

    # generate_report(config, df, _, _, res_dir_name)


def generate_report(config, metrics_df, mean_average_precision, mrr, results_path):
    with open("results/template.html") as f:
        report_html = f.read()
        report_html = report_html.replace("__DATA__", json.dumps(config))

        with open(config["model"]["config_path"]) as f2:
            model_config = json.load(f2)
        report_html = report_html.replace("__MODELDATA__", json.dumps(model_config))
        report_html = report_html.replace("__TABLE__", metrics_df.to_html())
        report_html = report_html.replace(
            "__MAP__", str(round(mean_average_precision, 3))
        )
        report_html = report_html.replace("__MRR__", str(round(mrr, 3)))

    with open(results_path + "/report.html", "w") as f:
        f.write(report_html)

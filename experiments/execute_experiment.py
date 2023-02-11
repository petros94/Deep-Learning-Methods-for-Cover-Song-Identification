import json
import os

import pandas as pd

from datasets.factory import make_dataset
from datetime import datetime
from models.classifier import ThresholdClassifier

from models.factory import make_model
from songbase.load_songs import from_config
from training.train import train
from utils.generic import split_songs
from utils.visualization import visualize_losses
from tuning.ranking_metrics import generate_metrics as generate_ranking_metrics, generate_embeddings_metrics
from tuning.classification_metrics import generate_metrics as generate_classification_metrics


def execute_single(config_path: str = "experiments/train_triplets.json"):
    print("Executing single experiment")
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Creating directory for checkpoint and saving configuration used")
    date_ = f"{datetime.now()}_{config['model']['type']}_{config['features']['input']}_{config['loss']}"
    chk_dir_name = config["train"]["checkpoints_path"] + date_
    os.mkdir(chk_dir_name)
    with open(config["model"]["config_path"]) as f2:
        model_config = json.load(f2)
    with open(chk_dir_name + "/config.json", "w") as f:
        json.dump(model_config, f)

    res_dir_name = config["train"]["results_path"] + date_
    os.mkdir(res_dir_name)

    print("Loading songs")
    train_songs, test_songs = from_config(config_path=config_path)

    train_songs, valid_songs = split_songs(train_songs, config["train"]["train_perc"])

    train_set = make_dataset(train_songs, config_path=config_path, type=config["loss"], segmented=True)
    valid_set = make_dataset(valid_songs, config_path=config_path, type=config["loss"], segmented=True)
    
    print(
        "Created training set: {} samples, valid set: {} samples".format(
            len(train_set), len(valid_set)
        )
    )

    model = make_model(config_path=config_path)

    print("Begin training")
    losses = train(
        model,
        train_set,
        valid_set,
        config_path=config_path,
        checkpoint_dir=chk_dir_name,
        results_dir=res_dir_name,
    )

    print("Plot losses")
    visualize_losses(losses, file_path=res_dir_name)
    
    if config['model']['type'] != "embeddings":
        evaluate_test_set(config_path, results_path=res_dir_name, test_songs=test_songs, valid_songs=valid_songs, model=model, segmented=True)
    evaluate_test_set(config_path, results_path=res_dir_name, test_songs=test_songs, valid_songs=valid_songs, model=model, segmented=False)
    
    return res_dir_name, chk_dir_name


def evaluate_model(config_path: str = "experiments/evaluation_pretrained.json", segmented=None):
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Creating directory for checkpoint and saving configuration used")
    date_ = f"{datetime.now()}_{config['model']['type']}_{config['features']['input']}_{config['loss']}"
    res_dir_name = config["train"]["results_path"] + date_
    os.mkdir(res_dir_name)

    print("Loading songs")
    _, test_songs = from_config(config_path=config_path)

    if segmented is None:
        evaluate_test_set(config_path=config_path, results_path=res_dir_name, test_songs=test_songs, segmented=True)
        evaluate_test_set(config_path=config_path, results_path=res_dir_name, test_songs=test_songs, segmented=False)
    else:
        evaluate_test_set(config_path=config_path, results_path=res_dir_name, test_songs=test_songs, segmented=segmented)

    return res_dir_name


def evaluate_test_set(config_path, results_path, test_songs, model=None, valid_songs=None, segmented=False):
    if segmented:
        print("Evaluating for segmented input")
        res_dir_name = results_path + '/segmented'
    else:
        print("Evaluating on full song input")    
        res_dir_name = results_path + '/full'
        
    os.mkdir(res_dir_name)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if len(test_songs) > 0:
        test_set = make_dataset(test_songs, config_path=config_path, type=config["loss"], segmented=segmented, n_batches=256)
    else:
        print("No test set provided, validation set will be used")
        test_set = make_dataset(valid_songs, config_path=config_path, type=config["loss"], segmented=segmented, n_batches=256)

    if model is None:
        model = make_model(config_path=config_path)
    
    print("Plot ROC and calculate metrics")
    if config["model"]["type"] == "embeddings":
        roc_stats, mean_average_precision, mrr, mr1, pr10 = generate_embeddings_metrics(model, test_set, results_path=res_dir_name)
    else:
        roc_stats, mean_average_precision, mrr, mr1, pr10 = generate_ranking_metrics(model, test_set, segmented=segmented, results_path=res_dir_name)

    print(f"MAP: {mean_average_precision}")
    print(f"MRR: {mrr}")
    print(f"MR1: {mr1}")
    print(f"Pr@10: {pr10}")

    try:
        thr = config["model"]["threshold"]
    except KeyError:
        thr = roc_stats.loc[roc_stats["tpr"] > 0.7, "thr"].iloc[0]

    clf = ThresholdClassifier(model, thr)

    if config["representation"] == ['wav']:
        df = pd.DataFrame()
    else:
        df = generate_classification_metrics(
            clf, test_set, segmented=segmented, results_path=res_dir_name
        )

    generate_report(config, df, mean_average_precision, mrr, res_dir_name)


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

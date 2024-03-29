import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score,
)
import plotly.express as px
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.nn import functional as F

from utils.generic import get_device
from training.miners import RandomTripletMiner
from datasets.TripletDataset import TripletDataset
from datasets.SimpleDataset import SimpleDataset


def generate_metrics(clf, data_set, segmented, results_path: str, balanced):
    if segmented:
        return generate_metrics_segments(clf, data_set, results_path)
    else:
        return generate_metrics_full(clf, data_set, results_path, balanced=balanced)


def generate_metrics_segments(
    clf, data_set: TripletDataset, results_path: str
):
    print(f"Classifier threshold: {clf.D}")
    clf.eval()
    device = get_device()
    clf.type(torch.FloatTensor).to(device)
    y_pred = []
    y_true = []
    miner = RandomTripletMiner
    with torch.no_grad():
        for i in range(len(data_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = data_set[i]
            data = data.type(torch.FloatTensor).to(device)

            a, p, n = miner(None, labels)

            pos_preds, _ = clf(data[a], data[p])
            pos_preds = pos_preds.cpu().tolist()
            y_pred.extend(pos_preds)
            y_true.extend([1] * len(pos_preds))

            neg_preds, _ = clf(data[a], data[n])
            neg_preds = neg_preds.cpu().tolist()
            y_pred.extend(neg_preds)
            y_true.extend([0] * len(neg_preds))

    return generate_metrics_bare(y_true, y_pred, results_path)


def generate_metrics_full(clf, data_set: SimpleDataset, results_path, balanced=True):
    clf.model.eval()
    device = get_device()
    clf.model.type(torch.FloatTensor).to(device)
    miner = RandomTripletMiner
    embeddings = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for frames, label in zip(data_set.frames, data_set.labels):
            x = frames.to(device)
            embeddings.append(clf.model(x))
            
        embeddings = torch.cat(embeddings, dim=0)

        if not balanced:
            distance_matrix = torch.cdist(embeddings, embeddings, p=2)
            song_labels = torch.tensor(data_set.labels)

            for id, (d, lab) in enumerate(zip(distance_matrix, song_labels)):
                ids = torch.argwhere(song_labels == lab)
                ids = ids.flatten()
                ids = ids[ids != id]
                pos_dist = d[ids]

                inv = 2 - pos_dist
                pos_preds = (inv > clf.D) * 1
                pos_preds = pos_preds.cpu().tolist()

                y_pred.extend(pos_preds)
                y_true.extend([1] * len(pos_preds))

                ids = torch.argwhere(song_labels != lab)
                ids = ids.flatten()
                neg_dist = d[ids]
                inv = 2 - neg_dist
                neg_preds = (inv > clf.D) * 1
                neg_preds = neg_preds.cpu().tolist()
                y_pred.extend(neg_preds)
                y_true.extend([0] * len(neg_preds))
        else:
            for i in range(100):
                a, p, n = miner(embeddings, torch.tensor(data_set.labels))

                pos_dist = torch.norm(embeddings[a] - embeddings[p], dim=1)
                inv = 2 - pos_dist
                pos_preds = (inv > clf.D)*1
                pos_preds = pos_preds.cpu().tolist()
                y_pred.extend(pos_preds)
                y_true.extend([1] * len(pos_preds))

                neg_dist = torch.norm(embeddings[a] - embeddings[n], dim=1)
                inv = 2 - neg_dist
                neg_preds = (inv > clf.D)*1
                neg_preds = neg_preds.cpu().tolist()
                y_pred.extend(neg_preds)
                y_true.extend([0] * len(neg_preds))
        
        return generate_metrics_bare(y_true, y_pred, results_path)


def generate_metrics_bare(y_true, y_pred, results_path):
    permute_ids = np.random.permutation(len(y_true))
    sample_y_true = np.array(y_true)[permute_ids][:100000]
    sample_y_pred = np.array(y_pred)[permute_ids][:100000]

    pr, rec, f1, sup = precision_recall_fscore_support(sample_y_true, sample_y_pred)
    acc = accuracy_score(sample_y_true, sample_y_pred)
    df = pd.DataFrame({"pre": pr, "rec": rec, "f1": f1, "sup": sup})
    ConfusionMatrixDisplay.from_predictions(sample_y_true, sample_y_pred)

    if results_path:
        df.to_csv(results_path + "/metrics.csv")
        plt.savefig(results_path + "/confusion.png")

    print(f"Accuracy is: {acc}")
    print(f"F1 is: {f1}")
    print(df)
    # plt.show()
    return df
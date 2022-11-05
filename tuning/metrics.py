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


def generate_metrics(clf, data_set, segmented, results_path: str):
    if segmented:
        return generate_metrics_segments(clf, data_set, results_path)
    else:
        return generate_metrics_full(clf, data_set, results_path)


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


def generate_metrics_full(clf, data_set: SimpleDataset, results_path):
    clf.model.eval()
    device = get_device()
    clf.model.type(torch.FloatTensor).to(device)
    
    embeddings = []
    with torch.no_grad():
        for frames, label in zip(data_set.frames, data_set.labels):
            x = frames.to(device)
            embeddings.append(clf.model(x))
            
        embeddings = torch.cat(embeddings, dim=0)
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        labels_matrix = torch.tensor([1*(lab_1 == lab_2) for lab_1 in data_set.labels for lab_2 in data_set.labels])
        
        distances, clf_labels = torch.flatten(distance_matrix).cpu().numpy(), torch.flatten(labels_matrix).cpu().numpy()
        inv = 2 - distances
        y_pred = (inv > clf.D)*1
        
        return generate_metrics_bare(clf_labels, y_pred, results_path)


def generate_metrics_bare(y_true, y_pred, results_path):
    pr, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    df = pd.DataFrame({"pre": pr, "rec": rec, "f1": f1, "sup": sup})
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    if results_path:
        df.to_csv(results_path + "/metrics.csv")
        plt.savefig(results_path + "/confusion.png")

    print(f"Accuracy is: {acc}")
    print(df)
    plt.show()
    return df


def mean_reprocical_rank(model, data_set, segmented):
    model.eval()
    device = get_device()
    model.type(torch.FloatTensor).to(device)
    with torch.no_grad():
        if segmented:
            for i in range(len(data_set)):
                # N X 1 X num_feats X num_samples, N
                (data, labels) = data_set[i]
                data = data.type(torch.FloatTensor).to(device)

                embeddings = model(data)

                # Size N x N
                rr = []
                distance_matrix = torch.cdist(embeddings, embeddings, p=2)
                for d, lab in zip(distance_matrix, labels):
                    sorted_ids_by_dist = d.argsort()
                    sorted_labels_by_dist = labels[sorted_ids_by_dist]

                    rank = 1
                    for test_label in sorted_labels_by_dist[1:]:
                        if lab == test_label:
                            break
                        rank += 1
                    rr.append(1 / rank)
            mrr = np.mean(rr)
            return mrr
        else:
            embeddings = []
            labels = data_set.labels
            for frames, label in zip(data_set.frames, data_set.labels):
                x = frames.to(device)
                embeddings.append(model(x))
            
            embeddings = torch.cat(embeddings, dim=0)
            # Size N x N
            rr = []
            distance_matrix = torch.cdist(embeddings, embeddings, p=2)
            
            print(distance_matrix.size())
            for d, lab in zip(distance_matrix, labels):
                sorted_ids_by_dist = d.argsort()
                sorted_labels_by_dist = labels[sorted_ids_by_dist]

                rank = 1
                for test_label in sorted_labels_by_dist[1:]:
                    if lab == test_label:
                        break
                    rank += 1
                rr.append(1 / rank)
            mrr = np.mean(rr)
            return mrr

import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
)
import plotly.express as px
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd
from torch.nn import functional as F

from utils.generic import get_device
from training.miners import RandomTripletMiner
from datasets.TripletDataset import TripletDataset
from datasets.SimpleDataset import SimpleDataset

def generate_ROC(model, data_set, segmented, results_path):
    if segmented:
        return generate_ROC_segments(model, data_set, results_path)
    else:
        return generate_ROC_full(model, data_set, results_path)
    
    
def generate_ROC_segments(
    model, data_set: TripletDataset, results_path: str
): 
    
    model.eval()
    device = get_device()
    model.type(torch.FloatTensor).to(device)
    distances = []
    clf_labels = []
    miner = RandomTripletMiner
    with torch.no_grad():
        for i in range(len(data_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = data_set[i]
            data = data.type(torch.FloatTensor).to(device)

            embeddings = model(data)
                
            a, p, n = miner(embeddings, labels)

            pos_dist = torch.norm(embeddings[a] - embeddings[p], dim=1)
            neg_dist = torch.norm(embeddings[a] - embeddings[n], dim=1)

            distances.append(pos_dist)
            clf_labels.extend([1] * pos_dist.size()[0])

            distances.append(neg_dist)
            clf_labels.extend([0] * neg_dist.size()[0])

    distances, clf_labels = torch.cat(distances).cpu().numpy(), np.array(clf_labels)
    return generate_ROC_bare(distances, clf_labels, results_path)


def generate_ROC_full(model, data_set: SimpleDataset, results_path):
    model.eval()
    device = get_device()
    model.type(torch.FloatTensor).to(device)
    miner = RandomTripletMiner
    embeddings = []
    distances = []
    clf_labels = []
    with torch.no_grad():
        for frames, label in zip(data_set.frames, data_set.labels):
            x = frames.to(device)
            embeddings.append(model(x))
            
        embeddings = torch.cat(embeddings, dim=0)
        
        for i in range(100):
            a, p, n = miner(embeddings, torch.tensor(data_set.labels))
            
            pos_dist = torch.norm(embeddings[a] - embeddings[p], dim=1)
            neg_dist = torch.norm(embeddings[a] - embeddings[n], dim=1)

            distances.append(pos_dist)
            clf_labels.extend([1] * pos_dist.size()[0])

            distances.append(neg_dist)
            clf_labels.extend([0] * neg_dist.size()[0])
            
       
        # distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        # labels_matrix = torch.tensor([1*(lab_1 == lab_2) for lab_1 in data_set.labels for lab_2 in data_set.labels])
        
        
        # distances, clf_labels = torch.flatten(distance_matrix).cpu().numpy(), torch.flatten(labels_matrix).cpu().numpy()
        distances, clf_labels = torch.cat(distances).cpu().numpy(), np.array(clf_labels)
        return generate_ROC_bare(distances, clf_labels, results_path)
    
            
def generate_ROC_bare(distances: np.ndarray, clf_labels: np.ndarray, results_path: str):
    fpr, tpr, thresholds = roc_curve(clf_labels, 2-distances)
    ap = average_precision_score(clf_labels, 2-distances)
    df = pd.DataFrame({"tpr": tpr, "fpr": fpr, "thr": thresholds})
    roc_auc = auc(fpr, tpr)

    fig = px.area(
        data_frame=df,
        x="fpr",
        y="tpr",
        title=f"ROC Curve (AUC={roc_auc:.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        hover_data=["thr"],
        width=700,
        height=500,
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")

    if results_path:
        df.to_csv(results_path + "/thresholds.csv")
        fig.write_image(results_path + "/roc.png")

    fig.show()
    return df, ap
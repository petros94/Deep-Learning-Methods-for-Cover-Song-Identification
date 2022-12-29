import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve
)
from sklearn.manifold import TSNE
import plotly.express as px
import torch
import pandas as pd

from utils.generic import get_device
from training.miners import RandomTripletMiner
from datasets.TripletDataset import TripletDataset
from datasets.SimpleDataset import SimpleDataset

def generate_metrics(model, data_set, segmented, results_path):
    if segmented:
        distances, clf_labels = generate_posteriors_segments(model, data_set)
    else:
        distances, clf_labels, embeddings, song_labels, cover_names = generate_posteriors_full(model, data_set)
        
    df = generate_ROC(distances, clf_labels, results_path)
    ap = average_precision(distances, clf_labels)
    mrr = mean_reprocical_rank(model, data_set, segmented)
    df_prc = generate_PRC(distances, clf_labels, results_path)

    if not segmented:
        generate_tsne(embeddings, song_labels, cover_names)

    return df, ap, mrr
    
    
def generate_posteriors_segments(
    model, data_set: TripletDataset): 
    
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
    return distances, clf_labels


def generate_posteriors_full(model, data_set: SimpleDataset):
    model.eval()
    device = get_device()
    model.type(torch.FloatTensor).to(device)
    miner = RandomTripletMiner
    embeddings = []
    song_labels = []
    cover_names = []
    distances = []
    clf_labels = []
    with torch.no_grad():
        for frames, song_label, cover_name in zip(data_set.frames, data_set.labels, data_set.song_names):
            x = frames.to(device)
            embeddings.append(model(x))
            song_labels.append(song_label)
            cover_names.append(cover_name)
            
        embeddings = torch.cat(embeddings, dim=0)

        for i in range(128):
            a, p, n = miner(embeddings, torch.tensor(data_set.labels))
            
            pos_dist = torch.norm(embeddings[a] - embeddings[p], dim=1)
            neg_dist = torch.norm(embeddings[a] - embeddings[n], dim=1)

            distances.append(pos_dist)
            clf_labels.extend([1] * pos_dist.size()[0])

            distances.append(neg_dist)
            clf_labels.extend([0] * neg_dist.size()[0])
            
        distances, clf_labels = torch.cat(distances).cpu().numpy(), np.array(clf_labels)
        return distances, clf_labels, embeddings, song_labels, cover_names
    
            
def generate_ROC(distances: np.ndarray, clf_labels: np.ndarray, results_path: str):
    fpr, tpr, thresholds = roc_curve(clf_labels, 2-distances)
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
    return df


def average_precision(distances: np.ndarray, clf_labels: np.ndarray):
    return average_precision_score(clf_labels, 2-distances)
    
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
                labels = labels.to(device)

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
            labels = torch.tensor(data_set.labels).to(device)
            for frames, label in zip(data_set.frames, data_set.labels):
                x = frames.to(device)
                embeddings.append(model(x))
            
            embeddings = torch.cat(embeddings, dim=0)
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


def generate_PRC(distances: np.ndarray, clf_labels: np.ndarray, results_path: str):
    # Use the precision_recall_curve function to get the precision, recall, and thresholds arrays
    precision, recall, thresholds = precision_recall_curve(clf_labels, 2-distances)
    df = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "thr": thresholds})
    # Compute the average precision score
    ap = average_precision_score(clf_labels, 2-distances)

    # Create a Plotly area plot using the precision, recall, and thresholds arrays
    fig = px.line(
        data_frame=df,
        x="recall",
        y="precision",
        title=f"Precision-Recall Curve (AP={ap:.4f})",
        labels=dict(x="Recall", y="Precision"),
        hover_data=["thr"],
        width=700,
        height=500,
    )

    # Update the y-axis to have the same scale as the x-axis
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # Constrain the x-axis to the [0, 1] domain
    fig.update_xaxes(constrain="domain")

    # Save the dataframe and the plot figure as a CSV file and a PNG image, respectively, if a results path is provided
    if results_path:
        fig.write_image(results_path + "/prc.png")

    # Display the plot
    fig.show()
    return df

def generate_tsne(embeddings: torch.Tensor, song_labels: list, cover_names: list) -> None:
    X = embeddings.cpu().numpy()
    X_embed = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=3).fit_transform(X)

    y = np.array(song_labels)
    fig = px.scatter(x=X_embed[:, 0], y=X_embed[:, 1], color=y, hover_name=cover_names)
    fig.show()
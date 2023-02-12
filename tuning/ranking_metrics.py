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
    distances, clf_labels, embeddings, song_labels, cover_names = generate_posteriors_full(model, data_set)
    df = generate_ROC(distances, clf_labels, results_path)
    df_prc = generate_PRC(distances, clf_labels, results_path)
    if segmented:
        return df, None, None, None, None
    else:
        map, mrr, mr1, pr10 = ranking_metrics(model, data_set, 10)
        generate_tsne(embeddings, song_labels, cover_names)
        return df, map, mrr, mr1, pr10


def generate_embeddings_metrics(model, data_set, results_path):
    class TempClass:
        def __init__(self, frames, labels, song_names):
            self.frames = frames.unsqueeze(1)
            self.labels = labels
            self.song_names = song_names

    embeddings, song_labels, cover_names = model.calculate_embeddings(data_set)

    distances, clf_labels, embeddings, song_labels, cover_names = generate_posteriors_full(model, TempClass(embeddings,
                                                                                                            song_labels,
                                                                                                            cover_names))
    df = generate_ROC(distances, clf_labels, results_path)
    ap = average_precision(distances, clf_labels)
    df_prc = generate_PRC(distances, clf_labels, results_path)
    generate_tsne(embeddings, song_labels, cover_names)
    return df, ap, pd.DataFrame(), None, None


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
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        song_labels = torch.tensor(song_labels)

        for id, (d, lab) in enumerate(zip(distance_matrix, song_labels)):
            ids = torch.argwhere(song_labels == lab)
            ids = ids.flatten()
            ids = ids[ids != id]
            pos_dist = d[ids]
            distances.append(pos_dist)
            clf_labels.extend([1] * pos_dist.size()[0])

            ids = torch.argwhere(song_labels != lab)
            ids = ids.flatten()
            neg_dist = d[ids]
            distances.append(neg_dist)
            clf_labels.extend([0] * neg_dist.size()[0])


        # for i in range(128):
        #     a, p, n = miner(embeddings, torch.tensor(data_set.labels))
        #
        #     pos_dist = torch.norm(embeddings[a] - embeddings[p], dim=1)
        #     neg_dist = torch.norm(embeddings[a] - embeddings[n], dim=1)
        #
        #     distances.append(pos_dist)
        #     clf_labels.extend([1] * pos_dist.size()[0])
        #
        #     distances.append(neg_dist)
        #     clf_labels.extend([0] * neg_dist.size()[0])

        distances, clf_labels = torch.cat(distances).cpu().numpy(), np.array(clf_labels)
        return distances, clf_labels, embeddings, song_labels, cover_names


def generate_ROC(distances: np.ndarray, clf_labels: np.ndarray, results_path: str):
    fpr, tpr, thresholds = roc_curve(clf_labels, 2 - distances)
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


def calc_MRR(distance_matrix, labels):
    # Size N x N
    rr = []

    for d, lab in zip(distance_matrix, labels):
        sorted_ids_by_dist = d.argsort()
        sorted_labels_by_dist = labels[sorted_ids_by_dist]

        # MRR
        rank = 1
        for test_label in sorted_labels_by_dist[1:]:
            if lab == test_label:
                break
            rank += 1
        rr.append(1 / rank)

    mrr = np.mean(rr)
    return mrr

def calc_MAP(array2d, version, que_range=None, K=1e10):
    if que_range is not None:
        que_s, que_t = que_range[0], que_range[1]
        if que_s == 0:
            ref_s, ref_t = que_t, len(array2d)
        else:
            ref_s, ref_t = 0, que_s
    else:
        que_s, que_t, ref_s, ref_t = 0, len(array2d), 0, len(array2d)

    new_array2d = []
    for u, row in enumerate(array2d[que_s: que_t]):
        row = [(v + ref_s, col) for (v, col) in enumerate(row[ref_s: ref_t]) if u + que_s != v + ref_s]
        new_array2d.append(row)
    MAP, top10, rank1 = 0, 0, 0

    for u, row in enumerate(new_array2d):
        row.sort(key=lambda x: x[1])
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = 0.
        u = u + que_s
        for k, (v, val) in enumerate(row):

            if version[u] == version[v]:

                if k < K:
                    version_cnt += 1
                    per_MAP += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1
        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        # if per_MAP < 0.1:
        #     print row
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1
    return MAP / float(que_t - que_s), top10 / float(que_t - que_s) / 10, rank1 / float(que_t - que_s)


def ranking_metrics(model, data_set, k):
    model.eval()
    device = get_device()
    model.type(torch.FloatTensor).to(device)
    with torch.no_grad():
        embeddings = []
        labels = torch.tensor(data_set.labels).to(device)
        for frames, label in zip(data_set.frames, data_set.labels):
            x = frames.to(device)
            embeddings.append(model(x))

        embeddings = torch.cat(embeddings, dim=0)
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        map, prk, mr1 = calc_MAP(distance_matrix, labels)
        mrr = calc_MRR(distance_matrix, labels)


        return map, mrr, mr1, prk


def generate_PRC(distances: np.ndarray, clf_labels: np.ndarray, results_path: str):
    # Use the precision_recall_curve function to get the precision, recall, and thresholds arrays
    precision, recall, thresholds = precision_recall_curve(clf_labels, 2 - distances)
    df = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "thr": thresholds})
    # Compute the average precision score
    ap = average_precision_score(clf_labels, 2 - distances)

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
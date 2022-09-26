import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_fscore_support, accuracy_score
import plotly.express as px
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd

from utils.generic import get_device
from training.miners import RandomTripletMiner

def generate_ROC(model, data_set: torch.utils.data.Dataset, batch_size: int, results_path: str):
    model.eval()
    device = get_device()
    model.to(device)
    distances = []
    clf_labels = []
    miner = RandomTripletMiner
    with torch.no_grad():
        for i in range(len(data_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = data_set[i]
            data = data.to(device)
            
            embeddings = model(data)
            a, p, n = miner(embeddings, labels)
            
            pos_dist = torch.norm(embeddings[a] - embeddings[p], dim=1)
            neg_dist = torch.norm(embeddings[a] - embeddings[n], dim=1)
            
            distances.append(pos_dist)
            clf_labels.extend([1]*pos_dist.size()[0])
            
            distances.append(neg_dist)
            clf_labels.extend([0]*neg_dist.size()[0])
            
    distances, clf_labels = torch.cat(distances).cpu().numpy(), np.array(clf_labels)
    fpr, tpr, thresholds = roc_curve(clf_labels, 1/distances)
    df = pd.DataFrame({'tpr': tpr, 'fpr': fpr, 'thr': thresholds})
    roc_auc = auc(fpr, tpr)
    
    fig = px.area(
        data_frame=df,
        x='fpr', y='tpr',
        title=f'ROC Curve (AUC={roc_auc:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        hover_data=['thr'],
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
        
    if results_path:
        df.to_csv(results_path + '/thresholds.csv')
        fig.write_image(results_path + '/roc.png')
        
    fig.show()
    return df
            

def generate_metrics(clf, data_set: torch.utils.data.Dataset, batch_size: int, results_path: str):
    print(f"Classifier threshold: {clf.D}")
    clf.eval()
    device = get_device()
    clf.to(device)
    y_pred = []
    y_true = []
    miner = RandomTripletMiner
    with torch.no_grad():
        for i in range(len(data_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = data_set[i]
            data = data.to(device)
            
            a, p, n = miner(None, labels)
            
            pos_preds, _ = clf(data[a], data[p])
            pos_preds = pos_preds.cpu().tolist()
            y_pred.extend(pos_preds)
            y_true.extend([1]*len(pos_preds))
            
            neg_preds, _ = clf(data[a], data[n])
            neg_preds = neg_preds.cpu().tolist()
            y_pred.extend(neg_preds)
            y_true.extend([0]*len(neg_preds))
                            
    pr, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    df = pd.DataFrame({'pre': pr, 'rec': rec, 'f1': f1, 'sup': sup})
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    
    if results_path:
        df.to_csv(results_path + '/metrics.csv')
        plt.savefig(results_path + '/confusion.png')
        
    print(f'Accuracy is: {acc}')
    print(df)
    plt.show()
    return df
        
        
        
    
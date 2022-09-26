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
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=getattr(data_set, "collate_fn", None))
    model.eval()
    device = get_device()
    model.to(device)
    distances = []
    labels = []
    miner = RandomTripletMiner
    with torch.no_grad():
        for i in range(len(data_set)):
            # N X 1 X num_feats X num_samples, N
            (data, labels) = data_set[i]
            data = data.to(device)
            
            embeddings = model(data)
            a, p, n = miner(embeddings, labels)
            
            emb_a, emb_p, emb_n = model(a), model(p), model(n)
            pos_dist = torch.norm(emb_a - emb_p, dim=1)
            neg_dist = torch.norm(emb_a - emb_n, dim=1)
            
            distances.append(pos_dist)
            labels.extend([1]*pos_dist.size()[0])
            
            distances.append(neg_dist)
            labels.extend([0]*neg_dist.size()[0])
            
        # for batch, (x, metadata) in enumerate(dataloader):     
        
        #     # x: 3 x N x 1 x W x H
        #     (anchor, pos, neg) = x 

        #     anchor.to(device)
        #     pos.to(device)
        #     neg.to(device)
            
        #     pair, label = ((anchor, pos), 1) if random.random() > 0.5 else ((anchor, neg), 0)
            
        #     #first dimension: N X 128
        #     first, second = model(pair[0]), model(pair[1])  
        #     dist = torch.norm(first - second, dim=1)       
        #     distances.append(dist)
        #     labels.extend([label]*dist.size()[0])
                            
    distances, labels = torch.cat(distances).cpu().numpy(), np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, 1/distances)
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
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=getattr(data_set, "collate_fn", None))
    clf.eval()
    device = get_device()
    clf.to(device)
    preds = []
    labels = []
    with torch.no_grad():
        for batch, (x, metadata) in enumerate(dataloader):     
        
            # x: 3 x N x 1 x W x H
            (anchor, pos, neg) = x 

            anchor.to(device)
            pos.to(device)
            neg.to(device)
            
            pair, label = ((anchor, pos), 1) if random.random() > 0.5 else ((anchor, neg), 0)
            
            #first dimension: N X 128
            pred, dist = clf(pair[0], pair[1])
            pred = pred.cpu().tolist()
            preds.extend(pred)
            labels.extend([label]*len(pred))
                            
    pr, rec, f1, sup = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    df = pd.DataFrame({'pre': pr, 'rec': rec, 'f1': f1, 'sup': sup})
    ConfusionMatrixDisplay.from_predictions(labels, preds)
    
    if results_path:
        df.to_csv(results_path + '/metrics.csv')
        plt.savefig(results_path + '/confusion.png')
        
    print(f'Accuracy is: {acc}')
    print(df)
    plt.show()
    return df
        
        
        
    
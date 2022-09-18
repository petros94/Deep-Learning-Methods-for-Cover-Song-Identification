import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd

from utils.generic import get_device

def generate_ROC(model, data_set: torch.utils.data.Dataset, batch_size: int, results_path: str):
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=getattr(data_set, "collate_fn", None))
    model.eval()
    device = get_device()
    model.to(device)
    distances = []
    labels = []
    with torch.no_grad():
        for batch, (x, metadata) in enumerate(dataloader):     
        
            # x: 3 x N x 1 x W x H
            (anchor, pos, neg) = x 

            anchor.to(device)
            pos.to(device)
            neg.to(device)
            
            pair, label = ((anchor, pos), 0) if random.random() > 0.5 else ((anchor, neg), 1)
            
            #first dimension: N X 128
            first, second = model(pair[0]), model(pair[1])  
            dist = torch.norm(first - second, dim=1)       
            distances.append(dist)
            labels.extend([label]*dist.size()[0])
                            
    distances, labels = torch.cat(distances).cpu().numpy(), np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    
    if results_path:
        df = pd.DataFrame({'tpr': tpr, 'fpr': fpr, 'thr': thresholds})
        df.to_csv(results_path + '/thresholds.csv')
        plt.savefig(results_path + '/roc.png')
        
        
    plt.show()
    

def generate_metrics(clf, data_set: torch.utils.data.Dataset, batch_size: int, results_path: str):
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
            
            pair, label = ((anchor, pos), 0) if random.random() > 0.5 else ((anchor, neg), 1)
            
            #first dimension: N X 128
            pred = clf(pair[0], pair[1]).cpu().tolist()  
            preds.extend(pred)
            labels.extend([label]*len(pred))
                            
    pr, rec, f1, sup = precision_recall_fscore_support(labels, preds)
    ConfusionMatrixDisplay.from_predictions(labels, preds)
    
    if results_path:
        pd.DataFrame({'pre': pr, 'rec': rec, 'f1': f1, 'sup': sup}).to_csv(results_path + '/metrics.csv')
        plt.savefig(results_path + '/confusion.png')
        
    plt.show()
    return pr, rec, f1, sup
        
        
        
    
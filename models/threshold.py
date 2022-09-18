import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import torch
import random

from utils.generic import get_device

class Threshold:
    def __init__(self, D) -> None:
        self.D = D
        
    def predict(self, distance):
        return distance < self.D
        
    def generate_ROC(self, model, data_set: torch.utils.data.Dataset, batch_size: int):
        dataloader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=getattr(data_set, "collate_fn", None))
        model.eval()
        device = get_device()
        model.to(device)
        output = []
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
                
                distances = torch.norm(first - second)
                print(distances.size())
                output.append(distances)
                labels.extend([label]*batch_size)
                
        print(len(output))
        print(output[0].size())
                
        output, labels = torch.cat(output).cpu().numpy(), np.array(labels)
        print(len(output))
        fpr, tpr, thresholds = roc_curve(labels, distances)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc[2],
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()
            
import torch 
import random

def RandomTripletMiner(embeddings, labels):
    anchors = []
    positives = []
    negatives = []
    for idx, label in enumerate(labels):
        
        anchor = idx
        pos_ids = torch.argwhere(labels == label).squeeze()
        pos = idx
        while pos == idx:
            pos = random.choice(pos_ids).item()
            
        neg_ids = torch.argwhere(labels != label).squeeze()
        neg = random.choice(neg_ids).item()
        
        anchors.append(anchor)
        positives.append(pos)
        negatives.append(neg)
        
    return (anchors, positives, negatives)
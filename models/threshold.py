import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

class Threshold:
    def __init__(self, D) -> None:
        self.D = D
        
    def predict(self, distance):
        return distance < self.D
        
    def generate_ROC(self, distances: np.array, labels: list):
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
            
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve, classification_report)

def evaluate_multi(y_true, y_pred):
    # evaluate multi-class classification in closedworld scenario => Monitored(0-94)
    # evaluate multi-class classification in openworld scenario => Monitored(0-94)+Unmonitored(-1)
    print("\n<Evaluation>")
    accuracy = accuracy_score(y_true, y_pred)
    # Macro Average
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    # Weighted Average
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

    # Classification Report
    report = classification_report(y_true, y_pred, zero_division=0)  # Avoid errors due to division by zero
    print("\nClassification Report:")
    print(report)
    
def evaluate_binary(y_true, y_pred, y_prob=None):
    # evaluate binary classification in openworld scenario => Monitored(1)+Unmonitored(-1)
    print("\n<Evaluation>")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Classification Report
    report = classification_report(y_true, y_pred, zero_division=0)  # Avoid errors due to division by zero
    print("\nClassification Report:")
    print(report)
    
    if y_prob is not None:
        auc_roc = roc_auc_score(y_true, y_prob[:, 1])
        print(f"AUC-ROC: {auc_roc:.4f}")

        # Precision-Recall Curve
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob[:, 1])
        plt.figure()
        plt.plot(recalls, precisions, label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(f"main_results/ow_binary_precision_recall_curve.png")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_roc:.4f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(f"main_results/ow_binary_roc_curve.png")
        plt.close()
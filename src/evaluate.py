import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
)


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
    
    if y_prob is not None:
        auc_roc = roc_auc_score(y_true, y_prob[:, 1])
        print(f"AUC-ROC: {auc_roc:.4f}")
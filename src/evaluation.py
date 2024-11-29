import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, precision_recall_curve, average_precision_score

def evaluate_multi(y_test, y_pred, y_proba, class_names, save_path_details, save_plots = False ):
    """
    Evaluates the classification model using various metrics, including Precision-Recall Curve,
    Confusion Matrix, and ROC Curve, and saves visualizations to files.
    
    Parameters:
    y_test (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_proba (array-like): Predicted probabilities (for ROC curve).
    class_names (list): List of class names for the confusion matrix.
    save_path (str): Path to save the plots.
    """
    import os

    save_path = "./data/plots"
    os.makedirs(save_path, exist_ok=True)
    
    # Accuracy, Precision, Recall, F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(save_path, f"${save_path_details}_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    print(f"Confusion matrix saved to {cm_path}")
    
    # TPR and FPR per class (ROC Curve)
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for random guessing
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", bbox_to_anchor=(1.05, 1), ncol=2) 
    roc_path = os.path.join(save_path, f"${save_path_details}_roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    print(f"ROC curve saved to {roc_path}")
    
    # Precision-Recall Curve
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test == i, y_proba[:, i])
        ap_score = average_precision_score(y_test == i, y_proba[:, i])
        plt.plot(recall, precision, label=f"Class {class_name} (AP = {ap_score:.2f})")
    
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", bbox_to_anchor=(1.05, 1), ncol=3)
    pr_path = os.path.join(save_path, f"${save_path_details}_precision_recall_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    print(f"Precision-Recall curve saved to {pr_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

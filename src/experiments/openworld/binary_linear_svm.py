import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load data
ow_path = '../../data/csv/openworld_binary_data.csv'
df = pd.read_csv(ow_path)

X = df.drop(columns=['Label'])  # 'Label' is the target
y = df['Label']

# Split into train, test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Parameters
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}  # Linear SVM parameters
k_values = ['all', 45, 40, 35]  # Feature selection values

# Results file path
results_file_path = "binary_linear_svm.txt"

# Ensure the file is clean before writing
with open(results_file_path, 'w') as file:
    file.write("Linear SVM Results with Stratified K-Fold\n")
    file.write("=" * 50 + "\n")

# SMOTE instance
smote = SMOTE(random_state=42)

# Stratified K-Fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Preprocess and optimize for each k
for k in k_values:
    print(f"\n--- Processing with k={k} features ---")

    # Feature selection
    if k == 'all':
        X_train_selected = X_train
        X_test_selected = X_test
    else:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

    # Apply SMOTE to the selected features
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

    # Scale the features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test_selected)

    best_params = None
    best_cv_accuracy = 0

    # Evaluate each parameter combination with Stratified K-Fold
    for C in param_grid['C']:
        fold_accuracies = []

        for train_idx, val_idx in kfold.split(X_train_scaled, y_train_resampled):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]

            # Train SVM model
            svm_model = SVC(kernel='linear', C=C, class_weight='balanced', random_state=42)
            svm_model.fit(X_fold_train, y_fold_train)

            # Validate model
            y_val_pred = svm_model.predict(X_fold_val)
            fold_accuracy = accuracy_score(y_fold_val, y_val_pred)
            fold_accuracies.append(fold_accuracy)

        # Calculate mean CV accuracy for this parameter
        mean_cv_accuracy = np.mean(fold_accuracies)

        # Check if this is the best parameter
        if mean_cv_accuracy > best_cv_accuracy:
            best_cv_accuracy = mean_cv_accuracy
            best_params = {'C': C}

        # Print intermediate results
        print(f"  C={C}, CV Accuracy={mean_cv_accuracy:.4f}")

    # Train final model with best parameters
    svm_final_model = SVC(kernel='linear', C=best_params['C'], class_weight='balanced', random_state=42)
    svm_final_model.fit(X_train_scaled, y_train_resampled)

    # Test the model on the test set
    y_test_pred = svm_final_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    classification_report_test = classification_report(y_test, y_test_pred, digits=4)

    # Record results for the current k
    with open(results_file_path, 'a') as file:
        file.write(f"\n=== Results for k={k} ===\n")
        file.write(f"Best Parameters: {best_params}\n")
        file.write(f"Best CV Accuracy: {best_cv_accuracy:.4f}\n")
        file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        file.write(f"Classification Report on Test Set:\n{classification_report_test}\n")
        file.write("=" * 50 + "\n")

    # Print final results for logging
    print(f"Best Parameters for k={k}: {best_params}")
    print(f"Best CV Accuracy: {best_cv_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Classification Report:\n{classification_report_test}")

print(f"\nAll results saved to {results_file_path}")

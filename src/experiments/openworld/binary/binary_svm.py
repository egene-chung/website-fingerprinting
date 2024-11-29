import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Load data
ow_path = '../../../data/csv/openworld_binary_data.csv'
df = pd.read_csv(ow_path)

X = df.drop(columns=['Label'])  # 'Label' is the target
y = df['Label']

# Split into train, test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Parameters
param_grid = {'C': [2048, 4096], 'gamma': [1, 0.5, 0.25]}  # Largest first
k_values = ['all', 45, 40, 35, 10]  # Largest first

# Initialize results storage
results = {}

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

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

    # Scale the features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test_selected)

    # Grid search with cross-validation
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    grid_search.fit(X_train_scaled, y_train_resampled)

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # Store results
    results[k] = {
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'classification_report': report
    }

    print(f"Best Parameters for k={k}: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")

# Save results to a file
results_file_path = "binary_svm.txt"
with open(results_file_path, 'w') as file:
    for k, result in results.items():
        file.write(f"\n=== Results for k={k} ===\n")
        file.write(f"Best Parameters: {result['best_params']}\n")
        file.write(f"Test Accuracy: {result['accuracy']:.4f}\n")
        file.write(f"Classification Report:\n{result['classification_report']}\n")
        file.write("=" * 50 + "\n")

print(f"\nResults saved to {results_file_path}")

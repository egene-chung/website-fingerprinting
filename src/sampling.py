from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.preprocess import check_and_preprocess_data
from evaluate import evaluate_multi, evaluate_binary

def train_randomforest(
    file_path, 
    scenario, 
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    max_features='sqrt', 
    selected_features=None,
    sampling_strategy="auto"  # 샘플링 전략 추가
):
    # load dataset
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"\nClass Distribution Before Sampling: \n{class_counts}")
    # Adjusted balance threshold
    imbalance_threshold = 0.8  # Defines the minimum acceptable ratio for balance

    if class_counts.min() / class_counts.max() < imbalance_threshold:  # Detect imbalance
        if class_counts.idxmin() == class_counts.index[-1]:  # If imbalance is caused by minority class
            print("\nApplying Oversampling (SMOTE)...")
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        else:
            print("\nApplying Undersampling (RandomUnderSampler)...")
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    else:
        print("\nClass distribution is close to balanced. No sampling applied.")
        sampler = None

    # data scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    # Apply sampling if needed
    if sampler:
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        print(f"\nClass Distribution After Sampling: \n{y_train.value_counts()}")

    # random forest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # train model before feature selection
    print(f"\n====== {scenario} ======")
    if selected_features:
        print("[Before Feature Selection]")

    model.fit(X_train, y_train)
    y_pred_all = model.predict(X_test)
    y_proba_all = model.predict_proba(X_test)
    
    # ==evaluation==
    if scenario == "Open World Binary":
        evaluate_binary(y_test, y_pred_all, y_proba_all)
    else:
        evaluate_multi(y_test, y_pred_all)
    
    # train with selected features (if provided)
    if selected_features:
        print("\n[After Feature Selection]")
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # re-train same model with selected features
        model.fit(X_train_selected, y_train)
        y_pred_selected = model.predict(X_test_selected)
        y_proba = model.predict_proba(X_test_selected)
        
        # ==evaluation==
        if scenario == "Open World Binary":
            evaluate_binary(y_test, y_pred_selected, y_proba)
        else:
            evaluate_multi(y_test, y_pred_selected)

def main():
    # file paths
    cw_path = 'data/csv/closedworld_data.csv'
    ow_binary_path = 'data/csv/openworld_binary_data.csv'
    ow_multi_path = 'data/csv/openworld_multi_data.csv'

    # check data/csv/ and check if these files exist
    check_and_preprocess_data(cw_path, ow_binary_path, ow_multi_path)
    
    # Best performing model + hyperparameters in scenarios
    train_randomforest(cw_path, scenario="Closed World", n_estimators=200, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", )
    train_randomforest(ow_binary_path, scenario="Open World Binary", n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="log2")
    train_randomforest(ow_multi_path, scenario="Open World Multi", n_estimators=200, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features="log2")

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data.preprocess import check_and_preprocess_data
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from evaluation import evaluate_multi, evaluate_binary

def train_rf_cw(X, y, X_train, X_test, y_train, y_test):
    # best performing model in the closed world scenario
    print("\n========Closed World(Random Forest)========")
    class_names = list(y.unique())
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # random forest model
    # these parameters were selected through iterative testing across various hyperparameter combinations.
    # can be found in /src/experiments/closedworld/cw_rf_backward_elimination.txt
    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="log2",
        bootstrap=False,
        criterion="gini",
        random_state=42
    )

    # ===Before Feature selection===
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    y_proba = rf_model.predict_proba(X_test_scaled)
    
    print("[Before Feature Selection]")
    evaluate_multi(y_test, y_pred, y_proba, class_names)

    # ===After Feature Selection===
    print("[After Feature Selection]")
    print("<Best Performance - through Forward Elimination in Random Forest>")
    # Feature Selection by applying Foward Elimination based on Feature Importance
    # Experiment in finding these features can be found in /src/experiments/closedworld/cw_rf_forward_elimination.txt
    selected_features = ['num_incoming_cum', 'std_total', 'num_incoming_thirty', 'std_dev_interval', 'std_outgoing_burst', 'std_total_cum', 'num_outgoing_last_thirty', 'avg_cum_incoming', 'std_timestamps', 'num_incoming_burst', 'num_outgoing_cum']

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    X_train_selected_scaled = pd.DataFrame(scaler.fit_transform(X_train_selected), columns=selected_features)
    X_test_selected_scaled = pd.DataFrame(scaler.transform(X_test_selected), columns=selected_features)

    # train model again after feature selection
    rf_model.fit(X_train_selected_scaled, y_train)
    y_pred = rf_model.predict(X_test_selected_scaled)
    y_proba = rf_model.predict_proba(X_test_selected_scaled)
    evaluate_multi(y_test, y_pred, y_proba, class_names, save_path_details="cw_rf", save_plots=True)

def train_svm_ow_binary(X, y, X_train, X_test, y_train, y_test):
    # best performing model in the open world binary scenario
    print("\n========Open World- Binary (SVM)========")
    num_features=40
    C_value=2048
    gamma_value=0.25

    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_selected = selector.fit_transform(X, y)
    
    # Scale the features to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Create an SVM classifier with the specified parameters
    svm_model = svm.SVC(kernel='rbf', C=C_value, gamma=gamma_value)

    # Train the SVM model on the training data
    svm_model.fit(X_train_scaled, y_train)

    # Use the trained classifier to predict labels for the test data
    y_pred = svm_model.predict(X_test_scaled)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

def train_svm_ow_binary(X, y, X_train, X_test, y_train, y_test, num_features=40, C_value=2048, gamma_value=0.25):
    print("\n========Open World- Binary (SVM)========")

    # 1. 처리되지 않은 데이터
    print("\n--- Using Raw Data ---")
    # Scale the features to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate SVM
    svm_model = svm.SVC(kernel='rbf', C=C_value, gamma=gamma_value)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    y_proba = svm_model.predict_proba(X_test_scaled)[:, 1] 

    evaluate_binary(y_test, y_pred, y_proba, save_plots=False)    

    # 2. SMOTE + SelectKBest 적용
    print("\n--- Using SMOTE + SelectKBest ---")
    # # Apply SMOTE
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Apply SelectKBest
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
    X_test_selected = selector.transform(X_test)

    # Scale the features
    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    # Train and evaluate SVM
    svm_model.fit(X_train_selected_scaled, y_train_resampled)
    y_pred_selected = svm_model.predict(X_test_selected_scaled)
    y_proba_selected = svm_model.predict_proba(X_test_selected_scaled)[:, 1]
    evaluate_binary(y_test, y_pred_selected, y_proba_selected, save_path_details="smote_selectkbest", save_plots="ow_binary")


def main():
    # 파일 경로
    cw_path = 'data/csv/closedworld_data.csv'
    ow_binary_path = 'data/csv/openworld_binary_data.csv'
    ow_multi_path = 'data/csv/openworld_multi_data.csv'

    check_and_preprocess_data(cw_path, ow_binary_path, ow_multi_path)

    # Best performing model and params in closed world
    df_cw = pd.read_csv(cw_path)
    X_cw = df_cw.drop(columns=['Label'])
    y_cw = df_cw['Label']

    X_train, X_test, y_train, y_test = train_test_split(X_cw, y_cw, test_size=0.2, stratify=y_cw, random_state=42)

    # train_rf_cw(X_cw, y_cw, X_train, X_test, y_train, y_test)
    
    # Best performing model and params in open world binary
    df_ow = pd.read_csv(ow_binary_path)
    X_ow = df_ow.drop(columns=['Label'])
    y_ow = df_ow['Label']

    # VarianceThreshold를 사용하여 상수 값 제거
    var_thresh = VarianceThreshold(threshold=0.0)  # 분산이 0인 feature 제거
    X_ow_reduced = var_thresh.fit_transform(X_ow)

    X_train, X_test, y_train, y_test = train_test_split(X_ow_reduced, y_ow, test_size=0.2, stratify=y_ow, random_state=42)
    
    train_svm_ow_binary(X_ow, y_ow, X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    main()
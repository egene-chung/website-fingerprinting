import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data.preprocess import check_and_preprocess_data
from evaluation import evaluate_multi

def train_rf_cw(X, y, X_train, X_test, y_train, y_test):
    # best performing model in the closed world scenario
    print("\n========Closed World(Random Forest)========")
    class_names = list(y.unique())
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # random forest model
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


    



def main():
    # 파일 경로
    ow_path = 'data/csv/openworld_data.csv'
    cw_path = 'data/csv/closedworld_data.csv'

    check_and_preprocess_data(ow_path, cw_path)

    df = pd.read_csv(cw_path)

    # Feature와 Label 정의
    X = df.drop(columns=['Label'])  # 'Label'이 타겟 컬럼
    y = df['Label']

    # 데이터셋 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_rf_cw(X, y, X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    main()
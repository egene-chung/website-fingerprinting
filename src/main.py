import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.preprocess import check_and_preprocess_data
from evaluate import evaluate_multi, evaluate_binary
from sampling import train_randomforest_with_sampling

def train_randomforest(
    file_path, 
    scenario, 
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    max_features='sqrt', 
    selected_features=None
):
    # load dataset
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # data scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
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
    print(f"\n====== {scenario}======")
    if selected_features:
        print("[Before Feature Selection]")

    model.fit(X_train, y_train)
    y_pred_all = model.predict(X_test)
    y_proba_all = model.predict_proba(X_test)
    
    # ==evaluation==
    if scenario == "Open World Binary" : 
        evaluate_binary(y_test,y_pred_all, y_proba_all)
    else : 
        evaluate_multi(y_test,y_pred_all)

    
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
        if scenario == "Open World Binary" : 
            evaluate_binary(y_test,y_pred_selected, y_proba)
        else : 
            evaluate_multi(y_test,y_pred_selected)

def main():
    # file paths
    cw_path = 'data/csv/closedworld_data.csv'
    ow_binary_path = 'data/csv/openworld_binary_data.csv'
    ow_multi_path = 'data/csv/openworld_multi_data.csv'

    # check data/csv/ and check if these files exist
    check_and_preprocess_data(cw_path, ow_binary_path, ow_multi_path)
    
    # cw_feature_selection in /feature_selection/cw_multi_forward.txt 
    # ow_binary_feature_selection in /feature_selection/ow_binary_backward.txt
    # ow_multi_feature_selection in /feature_selection/ow_multi_backward.txt
    cw_feature_selection = ['sum_incoming_outgoing_total_cum', 'fraction_incoming', 'num_incoming_thirty', 'range_interval', 'std_outgoing_burst', 'std_total_cum', 'num_incoming_last_thirty', 'avg_cum_incoming', 'avg_burst_incoming', 'std_timestamps', 'fraction_outgoing_burst', 'num_incoming_thirty_burst', 'num_outgoing', 'avg_total_cum', 'num_total', 'min_interval']
    ow_binary_feature_selection=['avg_outgoing', 'min_interval', 'num_incoming_last_thirty_cum', 'num_outgoing_last_thirty_cum', 'num_incoming_thirty_burst', 'num_outgoing_last_thirty_burst', 'num_incoming_last_thirty_burst', 'num_outgoing_thirty_burst', 'num_incoming_last_thirty', 'num_outgoing_last_thirty', 'fraction_outgoing_burst', 'num_total_cum', 'sum_incoming_outgoing_total_cum', 'num_incoming_burst', 'fraction_incoming_burst', 'sum_incoming_outgoing_total_burst_diff', 'sum_incoming_outgoing_total_burst', 'num_total_burst', 'num_outgoing_burst', 'num_total', 'num_incoming_cum', 'sum_incoming_outgoing_total', 'num_incoming', 'num_outgoing', 'std_outgoing_cum', 'avg_cum_incoming', 'fraction_incoming', 'fraction_incoming_burst_incoming_diff', 'avg_total_cum', 'fraction_outgoing', 'fraction_outgoing_burst_outgoing_diff', 'std_incoming_cum', 'std_total_cum', 'avg_total', 'std_total_burst', 'std_total', 'std_incoming_burst', 'mean_interval', 'std_dev_interval', 'range_interval', 'max_interval', 'avg_burst_incoming', 'std_timestamps', 'avg_total_burst', 'avg_timestamps', 'num_outgoing_thirty_cum', 'std_outgoing_burst', 'max_timestamps', 'avg_burst_outgoing', 'num_outgoing_cum', 'num_incoming_thirty', 'num_outgoing_thirty', 'num_incoming_thirty_cum', 'sum_incoming_outgoing_total_cum_diff']
    ow_multi_feature_selection = ['num_outgoing_last_thirty_cum', 'num_incoming_last_thirty_burst', 'num_incoming_thirty_burst', 'num_outgoing_last_thirty_burst', 'num_outgoing_thirty_burst', 'std_outgoing_cum', 'num_outgoing_thirty_cum', 'num_outgoing_cum', 'std_incoming_burst', 'avg_timestamps', 'fraction_incoming_burst', 'fraction_outgoing_burst', 'num_incoming_burst', 'num_outgoing_burst', 'std_total_burst', 'num_total_burst', 'sum_incoming_outgoing_total_burst', 'num_incoming_last_thirty', 'num_outgoing_last_thirty', 'fraction_outgoing', 'sum_incoming_outgoing_total_burst_diff', 'std_timestamps', 'std_total', 'sum_incoming_outgoing_total_cum', 'num_total_cum', 'fraction_incoming', 'num_total', 'num_incoming', 'num_incoming_cum', 'fraction_incoming_burst_incoming_diff', 'avg_total', 'fraction_outgoing_burst_outgoing_diff', 'sum_incoming_outgoing_total', 'mean_interval', 'avg_burst_incoming', 'avg_total_burst', 'avg_total_cum', 'avg_cum_incoming', 'range_interval', 'num_outgoing', 'max_interval', 'std_total_cum', 'std_dev_interval', 'std_incoming_cum', 'max_timestamps', 'avg_burst_outgoing', 'std_outgoing_burst', 'num_incoming_thirty', 'num_outgoing_thirty', 'sum_incoming_outgoing_total_cum_diff', 'num_incoming_thirty_cum']
    
    # best performing model + hyperparameters in scenarios (experiments in /main_experiments)
    # closed world
    train_randomforest(cw_path, scenario="Closed World", n_estimators=200, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", selected_features=cw_feature_selection)
    
    # open world binary (pre-fs, post-fs, sampling)
    train_randomforest(ow_binary_path, scenario="Open World Binary", n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="log2", selected_features=ow_binary_feature_selection)
    train_randomforest_with_sampling(ow_binary_path, scenario="Open World Binary", n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="log2")
    
    # open world multi (pre-fs, post-fs, sampling)
    train_randomforest(ow_multi_path, scenario="Open World Multi", n_estimators=200, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features="log2", selected_features=ow_multi_feature_selection)
    train_randomforest_with_sampling(ow_multi_path, scenario="Open World Multi", n_estimators=200, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features="log2")

if __name__ == "__main__":
    main()
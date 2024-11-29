import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from data.preprocess import check_and_preprocess_data
from sklearn.model_selection import cross_val_score
from evaluation import evaluate_multi

def train_random_forest(X, y, X_train, X_test, y_train, y_test):
    
    class_names = list(y.unique())
    
    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # X_test_scaled not used because

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

    threshold = 0.01

    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    y_proba = rf_model.predict_proba(X_test_scaled)  # Predicted probabilities
    
    evaluate_multi(y_test, y_pred, y_proba, class_names, save_path_details="Rf")

    # Feature Importance
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # print(feature_importances.to_string() + "\n")
    # selected_features = feature_importances[feature_importances > threshold].index
    selected_features = feature_importances.iloc[:-9].index


    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # 모델 학습
    rf_model.fit(X_train_selected, y_train)
    y_pred = rf_model.predict(X_test_selected)
    
    # 성능 평가
    # accuracy = accuracy_score(y_test, y_pred)
    # cross val
    # scores = cross_val_score(rf_model, X_train[selected_features], y_train, cv=5, scoring="accuracy")
    # print(f"Cross-Validation Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    class_names = list(y.unique())  # Replace with your actual class names
    y_proba = rf_model.predict_proba(X_test_selected)  # Predicted probabilities
    print("===Best Performance in Closed World Scenario===")
    print("After Backward Elimination in Random Forest")
    evaluate_multi(y_test, y_pred, y_proba, class_names, save_path_details="Rf_ft_selection")


# def train_

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

    train_random_forest(X, y, X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    main()
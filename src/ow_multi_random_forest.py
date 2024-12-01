import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 데이터 로드
cw_path = 'data/csv/openworld_multi_data.csv'
df = pd.read_csv(cw_path)

# Feature와 Label 정의
X = df.drop(columns=['Label'])  # 'Label'이 타겟 컬럼
y = df['Label']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified K-Fold 설정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 확장된 하이퍼파라미터 조합
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
}

# 모든 파라미터 조합 생성
param_combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_split'],
    param_grid['min_samples_leaf'],
    param_grid['max_features'],
))

# 결과 기록 파일 초기화
results_log = "ow_multi_random_forest.txt"
with open(results_log, "w") as log:
    log.write("Stratified K-Fold Performance (No Feature Importance)\n")
    log.write("=" * 80 + "\n")
    log.write("\n")

# 모든 파라미터 조합에 대해 실행
for params in param_combinations:
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = params
    
    # 결과 저장
    fold_accuracies = []
    
    # Stratified K-Fold 진행
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 모델 초기화
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42  # 재현 가능성을 위해 고정
        )

        # 모델 학습 및 평가
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)

    # 평균 Fold 결과 계산
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    # 결과 기록
    with open(results_log, "a") as log:
        log.write(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
                  f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                  f"max_features={max_features}\n")
        log.write(f"Mean Fold Accuracy: {mean_accuracy:.4f}\n")
        log.write("-" * 80 + "\n")

print(f"Results have been saved to {results_log}.")
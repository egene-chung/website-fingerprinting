import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 데이터 로드
cw_path = '../../data/csv/closedworld_data.csv'
df = pd.read_csv(cw_path)

# Feature와 Label 정의
X = df.drop(columns=['Label'])  # 'Label'이 타겟 컬럼
y = df['Label']

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 확장된 하이퍼파라미터 조합
param_grid = {
    'n_estimators': [50, 100, 200, 250, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# 모든 파라미터 조합 생성
param_combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_split'],
    param_grid['min_samples_leaf'],
    param_grid['max_features'],
    param_grid['bootstrap'],
    param_grid['criterion']
))

# 결과 기록 파일 초기화
results_log = "cw_random_forest.txt"
with open(results_log, "w") as log:
    log.write("Detailed Feature Importance and Threshold Performance\n")
    log.write("=" * 80 + "\n")
    log.write("\n")

# Feature Importance 기반 성능 평가
thresholds = [0.01, 0.02]  # Threshold: 0.01 ~ 0.20 (0.01 간격)

# 모든 파라미터 조합에 대해 실행
for params in param_combinations:
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, criterion = params
    
    # 모델 초기화
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        random_state=42  # 재현 가능성을 위해 고정
    )

    # 모델 학습
    rf_model.fit(X_train_scaled, y_train)

    # Feature Importance 계산
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Feature Importance 전체 출력
    with open(results_log, "a") as log:
        log.write(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, "
                  f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
                  f"max_features={max_features}, bootstrap={bootstrap}, criterion={criterion}\n")
        log.write("Feature Importance:\n")
        log.write(feature_importances.to_string() + "\n")
        log.write("=" * 80 + "\n")

    for threshold in thresholds:
        selected_features = feature_importances[feature_importances > threshold].index
        
        if len(selected_features) == 0:
            with open(results_log, "a") as log:
                log.write(f"  Threshold: {threshold:.2f}\n")
                log.write(f"  No features selected at this threshold.\n")
                log.write("-" * 50 + "\n")
            continue
        
        # 선택된 Feature로 데이터 필터링
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # 모델 학습
        rf_model.fit(X_train_selected, y_train)
        y_pred = rf_model.predict(X_test_selected)

        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, zero_division=0)

        # 결과 기록
        with open(results_log, "a") as log:
            log.write(f"  Threshold: {threshold:.2f}\n")
            log.write(f"  Selected Features ({len(selected_features)}): {list(selected_features)}\n")
            log.write(f"  Accuracy: {accuracy:.4f}\n")
            # log.write("  Classification Report:\n")
            # log.write(classification_rep + "\n")
            log.write("-" * 50 + "\n")

print(f"Results have been saved to {results_log}.")

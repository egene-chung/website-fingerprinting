import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 데이터 로드
cw_path = '../data/csv/openworld_binary_data.csv'  # 데이터 파일 경로
df = pd.read_csv(cw_path)

# Feature와 Label 정의
X = df.drop(columns=['Label'])  # 'Label'이 타겟 컬럼
y = df['Label']

# Data Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 데이터셋 분리
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 모든 Feature로 Random Forest 학습
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# 초기 정확도 (모든 Feature 사용)
accuracy_before_selection = accuracy_score(y_test, rf_model.predict(X_test_scaled))

# Feature Importance 추출
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)

# 결과 기록 파일 초기화
results_log = "ow_binary_backward.txt"
with open(results_log, "w") as log:
    log.write("Backward Elimination Results\n")
    log.write("=" * 80 + "\n")
    log.write(f"Accuracy Before Feature Selection: {accuracy_before_selection:.4f}\n")
    log.write("=" * 80 + "\n\n")

# backward elimination init
remaining_features = feature_importance_df['Feature'].tolist()  # sort by importance
best_accuracy = accuracy_before_selection  
best_features = remaining_features.copy()  # all features at init

# Backward Elimination
step = 1
while step <= 10 and len(remaining_features) > 1:
    # 가장 중요도가 낮은 특성 제거
    feature_to_remove = remaining_features.pop(0)
    current_features = remaining_features

    # 모델 재학습 및 평가
    rf_model.fit(X_train_scaled[current_features], y_train)
    y_pred = rf_model.predict(X_test_scaled[current_features])
    current_accuracy = accuracy_score(y_test, y_pred)

    # Best Accuracy 업데이트
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_features = current_features.copy() 

    # 결과 기록
    with open(results_log, "a") as log:
        log.write(f"Step {step}: Removed Feature: {feature_to_remove}\n")
        log.write(f"Remaining Features: {len(current_features)}\n")
        log.write(f"Accuracy: {current_accuracy:.4f}\n")
        log.write("=" * 80 + "\n\n")

    step += 1

# 최종 결과 기록
with open(results_log, "a") as log:
    log.write("Final Results\n")
    log.write("=" * 80 + "\n")
    log.write(f"Best Accuracy: {best_accuracy:.4f}\n")
    log.write(f"Accuracy Before Feature Selection: {accuracy_before_selection:.4f}\n")
    log.write(f"Final Selected Features: {best_features}\n")
    log.write("=" * 80 + "\n")

print(f"Results have been saved to {results_log}.")
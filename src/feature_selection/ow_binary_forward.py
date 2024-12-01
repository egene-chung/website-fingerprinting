import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 데이터 로드
cw_path = '../data/csv/openworld_binary_data.csv'
df = pd.read_csv(cw_path)

# Feature와 Label 정의
X = df.drop(columns=['Label'])  # 'Label'이 타겟 컬럼
y = df['Label']

# Data Scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 데이터셋 분리
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 결과 기록 파일 초기화
results_log = "ow_binary_forward.txt"
with open(results_log, "w") as log:
    log.write("Forward Elimination Results (Using Feature Importance)\n")
    log.write("=" * 80 + "\n")
    log.write("\n")

# Random Forest 모델 설정 (Feature Importance 계산용)
initial_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    random_state=42
)

# Feature Importance 계산
initial_rf_model.fit(X_train_scaled, y_train)
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": initial_rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Feature Importance 기반으로 Feature 순서 결정
remaining_features = feature_importances["Feature"].tolist()
selected_features = []
best_accuracy = 0
best_features_set = []

# 반복적으로 feature 추가
while remaining_features:
    performance_log = []

    for feature in remaining_features:
        # 현재 선택된 feature에 새로운 feature 추가
        current_features = selected_features + [feature]

        # 모델 학습
        initial_rf_model.fit(X_train_scaled[current_features], y_train)
        y_pred = initial_rf_model.predict(X_test_scaled[current_features])

        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        performance_log.append((feature, accuracy))

    # 가장 높은 accuracy를 제공하는 feature 선택
    best_feature, current_best_accuracy = max(performance_log, key=lambda x: x[1])

    # Feature 선택 조건
    if current_best_accuracy > best_accuracy:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_accuracy = current_best_accuracy
        best_features_set = selected_features[:]

        # 결과 기록
        with open(results_log, "a") as log:
            log.write(f"Selected Feature: {best_feature}\n")
            log.write(f"Current Features: {selected_features}\n")
            log.write(f"Accuracy: {current_best_accuracy:.4f}\n")
            log.write("-" * 50 + "\n")
    else:
        # 더 이상 성능 향상이 없으면 중지
        break

# 최종 결과 기록
with open(results_log, "a") as log:
    log.write("=" * 80 + "\n")
    log.write("Final Selected Features and Performance\n")
    log.write(f"Selected Features: {best_features_set}\n")
    log.write(f"Best Accuracy: {best_accuracy:.4f}\n")
    log.write("=" * 80 + "\n")

print(f"Results have been saved to {results_log}.")

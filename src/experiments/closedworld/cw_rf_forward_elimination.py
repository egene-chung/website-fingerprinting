import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# 결과 기록 파일 초기화
results_log = "cw_rf_forward_elimination.txt"
with open(results_log, "w") as log:
    log.write("Forward Elimination Results (Fixed Parameters)\n")
    log.write("=" * 80 + "\n")
    log.write("\n")

# Forward Elimination 초기화
remaining_features = list(X.columns)
selected_features = []
best_accuracy = 0
best_features_set = []

# Random Forest 모델 설정 (파라미터 고정)
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

# 반복적으로 feature 추가
while remaining_features:
    performance_log = []

    for feature in remaining_features:
        # 현재 선택된 feature에 새로운 feature 추가
        current_features = selected_features + [feature]

        # 모델 학습
        rf_model.fit(X_train_scaled[current_features], y_train)
        y_pred = rf_model.predict(X_test_scaled[current_features])

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

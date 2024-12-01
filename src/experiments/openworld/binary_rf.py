from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
import random

# 랜덤 시드 고정 (재현성을 위해)
random.seed(42)
np.random.seed(42)

# 데이터 로드
ow_path = '../../data/csv/openworld_binary_data.csv'
df = pd.read_csv(ow_path)

X = df.drop(columns=['Label'])  # 'Label'이 타겟 변수
y = df['Label']

# 학습 및 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 결과를 기록할 파일 설정
results_file_path = './binary_rf.txt'  # 랜덤 포레스트 결과 파일 경로
with open(results_file_path, 'w') as file:
    file.write("Open World Binary - Random Forest Experiment Results (No SMOTE)\n")
    file.write("=" * 50 + "\n")

# 피처 선택 값 설정: 큰 값부터 작은 값 순
k_values = [50, 40, 30, 20, 10]

# 정규화 및 다항 특징 생성 설정
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# 랜덤 포레스트 하이퍼파라미터 설정
param_combinations = [
    {'n_estimators': n, 'max_depth': d, 'min_samples_split': s, 'min_samples_leaf': l, 'bootstrap': b}
    for n in [100, 200, 300]
    for d in [None, 10, 20]
    for s in [2, 5]
    for l in [1, 2]
    for b in [True, False]
]

# Stratified K-Fold 설정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 실험 루프
for k in k_values:
    # 피처 선택
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    for normalize in [True, False]:
        for feature_eng in [True, False]:
            # 정규화 적용 여부
            if normalize:
                scaler.fit(X_train_selected)
                X_train_trans = scaler.transform(X_train_selected)
                X_test_trans = scaler.transform(X_test_selected)
            else:
                X_train_trans = X_train_selected
                X_test_trans = X_test_selected

            # 피처 엔지니어링 적용 여부 (다항 특징 생성)
            if feature_eng:
                X_train_trans = poly.fit_transform(X_train_trans)
                X_test_trans = poly.transform(X_test_trans)

            for params in param_combinations:
                # 랜덤 포레스트 모델 초기화
                rf = RandomForestClassifier(**params, random_state=42)

                # Stratified K-Fold 교차 검증 수행
                cv_scores = []
                for train_idx, val_idx in skf.split(X_train_trans, y_train):
                    X_cv_train, X_cv_val = X_train_trans[train_idx], X_train_trans[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    rf.fit(X_cv_train, y_cv_train)
                    y_cv_pred = rf.predict(X_cv_val)
                    cv_scores.append(accuracy_score(y_cv_val, y_cv_pred))

                # 교차 검증 평균 및 표준편차 계산
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)

                # 테스트 세트 평가
                rf.fit(X_train_trans, y_train)
                y_test_pred = rf.predict(X_test_trans)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                # 결과 기록
                result = {
                    'Feature_Select_k': k,
                    'Normalize': normalize,
                    'Feature_Engineering': feature_eng,
                    'Params': params,
                    'Cross_Val_Mean_Accuracy': mean_cv_score,
                    'Cross_Val_Std': std_cv_score,
                    'Test_Accuracy': test_accuracy
                }
                
                # 파일에 기록
                with open(results_file_path, 'a') as file:
                    file.write(str(result) + '\n')

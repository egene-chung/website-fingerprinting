from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
import random

# 랜덤 시드 고정 (재현성을 위해)
random.seed(42)
np.random.seed(42)


# 데이터 로드
ow_path = '../../../data/csv/openworld_data.csv'
df = pd.read_csv(ow_path)

X = df.drop(columns=['Label'])  # 'Label'이 타겟 변수
y = df['Label']

# 학습 및 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 클래스 불균형 처리 (SMOTE 적용)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 결과를 기록할 파일 설정
results_file_path = './binary_rf_results.txt'  # 랜덤 포레스트 결과 파일 경로
with open(results_file_path, 'w') as file:
    file.write("Open World Binary - Random Forest Experiment Results (SMOTE Applied)\n")
    file.write("=" * 50 + "\n")

# 피처 선택 및 실험 설정
k_values = [10, 15, 20, 25, 30, 35, 40, 44, 47, 50]  # SelectKBest에서 사용할 k 값
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# 랜덤 포레스트 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 실험 루프
for k in k_values:
    # 피처 선택
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_balanced, y_train_balanced)
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

            # 랜덤 포레스트 모델 및 Grid Search 설정
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_trans, y_train_balanced)

            # 최적 모델 선택 및 평가
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_trans)
            accuracy = accuracy_score(y_test, y_pred)
            cross_val_scores = cross_val_score(
                best_model, X_train_trans, y_train_balanced, cv=5, scoring='accuracy', n_jobs=-1
            )

            # 결과 기록
            result = {
                'Feature_Select_k': k,
                'Normalize': normalize,
                'Feature_Engineering': feature_eng,
                'Best_Params': grid_search.best_params_,
                'Test_Accuracy': accuracy,
                'Cross_Val_Mean_Accuracy': np.mean(cross_val_scores),
                'Cross_Val_Std': np.std(cross_val_scores)
            }
            
            # 파일에 기록
            with open(results_file_path, 'a') as file:
                file.write(str(result) + '\n')

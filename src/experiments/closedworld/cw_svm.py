from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import datetime
import itertools

# 데이터 로드
cw_path = '../../data/csv/closedworld_data.csv'
df = pd.read_csv(cw_path)

# Feature와 Label 정의
X = df.drop(columns=['Label'])  # 'Label'이 타겟 컬럼
y = df['Label']

# 데이터셋 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection 기준값들 ('all' 포함)
k_values = [50, 40, 35, 30, 25, 'all']

# SVM 파라미터 그리드
param_grid = {'C': [2048, 4096, 8192, 16384, 32768, 65536, 131072],
              'gamma': [0.001, 0.01, 0.05, 0.1, 0.125, 0.25, 0.5, 1, 2, 4, 8]}

# 모든 조합 생성
param_combinations = list(itertools.product(param_grid['C'], param_grid['gamma']))

# 결과 기록 파일 경로
results_file_path = './cw_svm.txt'

# 현재 시간 기록
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 로그 파일 초기화
with open(results_file_path, 'a') as file:
    file.write(f"\nSVM Feature Selection Experiment Results (Accuracy Only) - {current_time}\n")
    file.write("=" * 50 + "\n")

# SelectKBest 기반 Feature Selection 및 SVM 학습
for k in k_values:
    if k == 'all':
        # Feature Selection 생략
        X_train_selected = X_train
        X_test_selected = X_test
    else:
        # SelectKBest 적용
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    
    # 모든 조합에 대해 반복
    for C, gamma in param_combinations:
        # SVM 모델 생성
        svm_model = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, 
                        class_weight='balanced', random_state=42)
        svm_model.fit(X_train_selected, y_train)
        
        # 테스트 데이터 평가
        y_pred = svm_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 결과 기록
        with open(results_file_path, 'a') as file:
            file.write(f"Features Selected: {k}, C={C}, gamma={gamma}\n")
            file.write(f"Test Accuracy: {accuracy:.4f}\n")
            file.write("=" * 50 + "\n")
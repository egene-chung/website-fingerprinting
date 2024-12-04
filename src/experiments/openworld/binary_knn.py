from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd

# load data
ow_path = '../../data/csv/openworld_binary_data.csv'
df = pd.read_csv(ow_path)

X = df.drop(columns=['Label'])  # 'Label' is target
y = df['Label']

# split into train, test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Class imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# log results
results_file_path = './binary_knn.txt'  # open world
with open(results_file_path, 'w') as file:
    file.write("Open world binary - KNN Experiment Results (SMOTE Applied)\n")
    file.write("=" * 50 + "\n")

# Feature Selection 및 실험
k_values = [10, 15, 20, 25, 30]  # SelectKBest에서 사용할 k 값
scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 25, 30, 35, 40],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

for k in k_values:
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

            # Feature Engineering 적용 여부
            if feature_eng:
                X_train_trans = poly.fit_transform(X_train_trans)
                X_test_trans = poly.transform(X_test_trans)

            # Grid Search & 모델 학습
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train_trans, y_train_balanced)

            # 최적 모델 및 평가
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_trans)
            accuracy = accuracy_score(y_test, y_pred)
            cross_val_scores = cross_val_score(best_model, X_train_trans, y_train_balanced, cv=5, scoring='accuracy')

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

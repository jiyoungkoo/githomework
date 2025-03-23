# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

X = wine.data  
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


####### A 작업자 작업 수행 #######

dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=4, random_state=42)

# 모델 학습
dt_model.fit(X_train, y_train)

# 예측
y_pred_dt = dt_model.predict(X_test)

# 성능 평가 (정확도)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# 결과 출력
print(f"Decision Tree Accuracy: {accuracy_dt}")


####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''



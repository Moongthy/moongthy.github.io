---
title: 2 랜덤 포레스트
author: mng
date: 2022-06-02 06:01:00 +0900
categories: [RANDOM FOREST]
tags: [random forest]
---

랜덤 포레스트의 작동 방식을 잘 이해하기 위해 사이킷-런으로 직접 모델-링을 해볼것이다.

# 2.1 랜덤 포레스트 분류 모델

---

간단한 인구조사 데이터셋으로 연봉이 5만달러 이상인지 예측하는 랜덤 포레스트 분류기를 만들어 본다. cross_val_score() 함수를 통해 테스트 결과가 잘 일반화 되는 지 확인해 볼 것이다.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 간편하게 받을 수 있는 인구 조사 데이터
df_census = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                        header=None)
df_census.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'capital-gain', 'capital-loss', 'hours-per-week',
                     'native-country', 'income']
df_census.drop(['education'], axis=1, inplace=True) # 일단 버려
df_census = pd.get_dummies(df_census) # 원핫 인코딩
df_census.drop(['income_ <=50K'], axis=1, inplace=True) # 타깃이 쪼개졌으므로 하나 지워준다

# feature와 target을 분리해줘야 한다
X_census = df_census.iloc[:, :-1]
y_census = df_census.iloc[:, -1]

# 모델링
rf = RandomForestClassifier(n_estimators=10, random_state=2, n_jobs=-1)

scores = cross_val_score(rf, X_census, y_census, cv=5)

print(f'accuracy:{np.round(scores, 3)}')
print(f'accuracy mean:{scores.mean():.3f}')
```

accuracy:[0.851 0.844 0.851 0.852 0.851]
accuracy mean:0.850

기본 랜덤 포레스트 분류기가 결정트리 모델 (81%)보다 인구조사 데이터셋에서 더 나은 점수를 만든다.

성능이 향상된 것은 배깅 때문일 것이다. 이 랜덤 포레스트는 10개의 트리(n_estimators=10)를 사용하기 때문에 한 개가 아니라 10개의 결정 트리를 기반으로 예측을 만든다. 각 트리는 부트 스트래핑 샘플을 사용하므로 다양성이 높아지고 이를 집계하면 분산이 줄어든다.

기본적으로 랜덤 포레스트 분류기는 노드를 분할 할 때
<span style="color:orange">
feature개수의 제곱근
</span>
을 사용한다. 예를 들어 100개의 특성이 있다면 랜덤 포레스트의 각 결정 트리는 10개의 특성만 사용한다. 따라서 중복 샘플을 가진 두 트리의 분할이 달라지기 때문에 매우 다른 예측을 만들 수 있다. 이것이 랜덤 포레스트가 분산을 줄이는 또 하나의 방법이다.

# 2.2 랜덤 포레스트 회귀 모델

---

랜덤 포레스트 회귀 모델은 분류 모델과 마찬가지로 부트스트랩 샘플을 사용하지만
노드 분할에 feature 개수의 제곱근이 아니라
<span style="color:orange">
feature 전체
</span>
를 사용한다.

최종 예측은 다수결 투표가 아니라 모든 트리의 예측을 평균하여 만든다.

```python
df_bikes = pd.read_csv('./bike_rentals_cleaned.csv')
print(df_bikes.head())

X_bikes = df_bikes.iloc[:, :-1]
y_bikes = df_bikes.iloc[:, -1]

# 모델링
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10, random_state=2, n_jobs=-1)

scores = cross_val_score(rf, X_bikes, y_bikes, scoring='neg_mean_squared_error',
                         cv=10)

rmse = np.sqrt(-scores)
print(f'RMSE:{np.round(rmse, 3)}')
print(f'RMSE mean:{rmse.mean():.3f}')
```

RMSE:[ 801.486  579.987  551.347  846.698  895.05  1097.522  893.738  809.284
833.488 2145.046]
RMSE mean:945.365

이 랜덤 포레스트 모델은 이전에 하이퍼 파라미터 튜닝을 해줬던 단일 결정 트리 모델 만큼은 아니지만 잘 수행된다. 이에 대한 이유는 자전거 대여 데이터셋을 통해 자세히 알아볼 것이다.

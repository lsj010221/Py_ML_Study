# 사이킷런으로 시작하는 머신러닝

## 목차

- [Iris 예제](#iris-예제)
- [Modules](#modules)
  - [train_test_split()](#traintestsplit)
  - [KFold()](#kfold)
  - [StratifiedKFolde()](#stratifiedkfolde)
  - [cross_val_score()](#crossvalscore)
- [Encoding](#encoding)
  - [Label Encoding](#label-encoding)
  - [One-Hot Encoding](#one-hot-encoding)
- [Feature Scaling](#feature-scaling)
  - [StandardScaler](#standardscaler)
  - [MinMaxScaler](#minmaxscaler)

## Iris 예제

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_label = iris.target

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df)
iris_df.to_csv('iris_df.csv',sep=',',na_rep='NaN')
X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_label, test_size=0.2)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,Y_train)

pred = dt_clf.predict(X_test)
print(accuracy_score(Y_test,pred))
```

```
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  label
0                  5.1               3.5                1.4               0.2      0
1                  4.9               3.0                1.4               0.2      0
2                  4.7               3.2                1.3               0.2      0
3                  4.6               3.1                1.5               0.2      0
4                  5.0               3.6                1.4               0.2      0
..                 ...               ...                ...               ...    ...
145                6.7               3.0                5.2               2.3      2
146                6.3               2.5                5.0               1.9      2
147                6.5               3.0                5.2               2.0      2
148                6.2               3.4                5.4               2.3      2
149                5.9               3.0                5.1               1.8      2

[150 rows x 5 columns]
0.9333333333333333
```

Iris 외에도 내장된 예제 데이터 세트 존재

boston, breast_cancer, digits 등

## Modules

### train_test_split()

학습 데이터 세트를 이용해 학습을 진행한 후 다시 학습 데이터를 이용해 예측을 할 경우 예측 성능을 정확히 측정할 수 없음. 따라서 학습과 예측에 사용되는 데이터를 분리해줄 필요가 있음.

```python
train_test_split(iris_data, iris_label, test_size=0.2)
train_test_split(iris_data, iris_label, train_size=0.2, random_state=12345)
train_test_split(iris_data, iris_label, shuffle=False)
```

test_size : 전체 데이터 세트 중 테스트 데이터 세트의 크기 (default 0.25)

train_size : 전체 데이터 세트 중 테스트 데이터 세트의 크기

shuffle : 분리 전 데이터를 미리 섞어 데이터를 분산 (default True)

random_state : 데이터 세트 생성 시 사용되는 난수 (default random)

### KFold()

K 폴드

가장 보편적으로 사요되는 교차 검증 기법

데이터 세트를 K개로 나누어 총 k번의 학습과 예측 시행

각 시행에서 k번째의 데이터 세트를 예측에, 나머지 데이터 세트를 학습에 사용

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier()

kfold = KFold(n_splits=5,shuffle=True)

for train, test in kfold.split(features) :
    X_train = features[train]
    X_test = features[test]
    Y_train = label[train]
    Y_test = label[test]
    
    dt_clf.fit(X_train,Y_train)
    pred = dt_clf.predict(X_test)
    print(accuracy_score(Y_test,pred))
```

```
0.9666666666666667
0.9333333333333333
0.9333333333333333
0.9333333333333333
0.9666666666666667
```

### StratifiedKFolde()

Stratified K 폴드

레이블 값들의 빈도가 불균형한 데이터 세트를 위한 K 폴드

데이터 세트를 나눌 때 전체 데이터 셋의 레이블 값들의 비율을 유지

KFold의 shuffle=true는 단순 랜덤이므로 비율이 일정하지 않을 수 있음

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier()

skf = StratifiedKFold(n_splits=5)

for train, test in skf.split(features,label) :
    print(test)
    X_train = features[train]
    X_test = features[test]
    Y_train = label[train]
    Y_test = label[test]
    
    dt_clf.fit(X_train,Y_train)
    pred = dt_clf.predict(X_test)
    print(accuracy_score(Y_test,pred))
```

```
[  0   1   2   3   4   5   6   7   8   9  50  51  52  53  54  55  56  57
  58  59 100 101 102 103 104 105 106 107 108 109]
0.9666666666666667
[ 10  11  12  13  14  15  16  17  18  19  60  61  62  63  64  65  66  67
  68  69 110 111 112 113 114 115 116 117 118 119]
0.9666666666666667
[ 20  21  22  23  24  25  26  27  28  29  70  71  72  73  74  75  76  77
  78  79 120 121 122 123 124 125 126 127 128 129]
0.9
[ 30  31  32  33  34  35  36  37  38  39  80  81  82  83  84  85  86  87
  88  89 130 131 132 133 134 135 136 137 138 139]
0.9666666666666667
[ 40  41  42  43  44  45  46  47  48  49  90  91  92  93  94  95  96  97
  98  99 140 141 142 143 144 145 146 147 148 149]
1.0
```

shuffle=True를 하지 않았음에도 테스트 인덱스가 레이블 값들의 비율을 유지하도록 지정된것을 확인할 수 있음

### cross_val_score()

교차 검증을 간편하게 할 수 있게 해주는 API

기본 Stratified K 폴드 방식으로 분할

회귀 등의 Stratified K 폴드가 불가능할 경우 K 폴드 방식으로 분할

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier()

score = cross_val_score(dt_clf, features, label, scoring='accuracy', cv=3)
print(score)
```

```
[0.98 0.92 0.98]
```

비슷한 API로 여러개의 평가 지표를 반환할 수 있는 cross_validate() 존재

## Encoding

ML 알고리즘은 문자열 값을 입력값으로 혀용하지 않음
따라서 인코딩을 통해 문자열 값을 숫자로 변환해주는 과정이 필요함

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

items = ['바랄게', '뭐', '더', '있어', '한여름밤의', '꿀', '한여름밤의', '꿀', 'so', 'sweet', 'so', 'sweet']

encoder = LabelEncoder()
encoder.fit(items)

print(encoder.transform(items))
print(encoder.classes_)
print(encoder.inverse_transform([2,0,1,3,6]))
```

```
[5 4 3 6 7 2 7 2 0 1 0 1]
['so' 'sweet' '꿀' '더' '뭐' '바랄게' '있어' '한여름밤의']
['꿀' 'so' 'sweet' '더' '있어']
```

### One-Hot Encoding

Label Encoding의 경우 문자열 값을 서로 다른 수로 표현

따라서 ML 알고리즘에 따라 가중치로 인식할 수 있음

이러한 문제점을 해결하기위해 One-Hot Encoding 사용

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

items = ['바랄게', '뭐', '더', '있어', '한여름밤의', '꿀', '한여름밤의', '꿀', 'so', 'sweet', 'so', 'sweet']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
labels = labels.reshape(-1,1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)

print(oh_labels.toarray())
print(oh_labels.shape)
```

```
[[0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]]
(12, 8)
```

One-Hot Encoding은 레이블을 2차원으로 인코딩

레이블 값에 따라 새로운 피처를 추가해 해당되는 값에만 1, 나머지 값들에는 0을 표시

Pandas에는 One-Hot Encoding을 더 간편하게 할 수 있게 해주는 API pd.get_dummies() 존재

```python
import pandas as pd

df = pd.DataFrame({'item':['바랄게', '뭐', '더', '있어', '한여름밤의', '꿀', '한여름밤의', '꿀', 'so', 'sweet', 'so', 'sweet']})
print(pd.get_dummies(df))
```

```
    item_so  item_sweet  item_꿀  item_더  item_뭐  item_바랄게  item_있어  item_한여름밤의
0         0           0       0       0       0         1        0           0
1         0           0       0       0       1         0        0           0
2         0           0       0       1       0         0        0           0
3         0           0       0       0       0         0        1           0
4         0           0       0       0       0         0        0           1
5         0           0       1       0       0         0        0           0
6         0           0       0       0       0         0        0           1
7         0           0       1       0       0         0        0           0
8         1           0       0       0       0         0        0           0
9         0           1       0       0       0         0        0           0
10        1           0       0       0       0         0        0           0
11        0           1       0       0       0         0        0           0
```

### Feature Scaling

피처 스케일링(Feature Scaling) : 서로 다른 변수의 값들을 일정한 수준으로 맞추는 작업

대표적인 방법인 표준화(Standardization)와 정규화(Normalization)

사이킷런에서는 대표적인 Feature Scaling 클래스인 StandardScaler와 MinMaxScaler 제공

### StandardScaler

데이터의 피처 각각이 평균이 0이고 분산이 1인 가우시안 정규분포를 가진 값으로 변환

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_label = iris.target
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
iris_scaled_df = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)

print(iris_df)
print(iris_scaled_df)
```

```
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8

[150 rows x 4 columns]
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0            -0.900681          1.019004          -1.340227         -1.315444
1            -1.143017         -0.131979          -1.340227         -1.315444
2            -1.385353          0.328414          -1.397064         -1.315444
3            -1.506521          0.098217          -1.283389         -1.315444
4            -1.021849          1.249201          -1.340227         -1.315444
..                 ...               ...                ...               ...
145           1.038005         -0.131979           0.819596          1.448832
146           0.553333         -1.282963           0.705921          0.922303
147           0.795669         -0.131979           0.819596          1.053935
148           0.432165          0.788808           0.933271          1.448832
149           0.068662         -0.131979           0.762758          0.790671

[150 rows x 4 columns]
```

### MinMaxScaler

데이터의 피처 각각이 0부터 1, 또는 -1부터 1 사이의 값으로 변환

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_label = iris.target
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
iris_scaled_df = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)

print(iris_df)
print(iris_scaled_df)
```

```
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8

[150 rows x 4 columns]
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0             0.222222          0.625000           0.067797          0.041667
1             0.166667          0.416667           0.067797          0.041667
2             0.111111          0.500000           0.050847          0.041667
3             0.083333          0.458333           0.084746          0.041667
4             0.194444          0.666667           0.067797          0.041667
..                 ...               ...                ...               ...
145           0.666667          0.416667           0.711864          0.916667
146           0.555556          0.208333           0.677966          0.750000
147           0.611111          0.416667           0.711864          0.791667
148           0.527778          0.583333           0.745763          0.916667
149           0.444444          0.416667           0.694915          0.708333

[150 rows x 4 columns]
```
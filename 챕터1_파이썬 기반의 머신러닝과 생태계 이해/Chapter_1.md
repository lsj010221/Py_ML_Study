# 파이썬 기반의 머신러닝과 생태계 이해

## 목차

- [머신러닝의 분류](#머신러닝의-분류)
- [개발 환경 세팅](#개발-환경-세팅)
- [Numpy](#numpy)
  - [import](#import)
  - [numpy.array()](#numpyarray)


## 머신러닝의 분류

- 지도학습 (Supervised Learning)

  정답이 있는 데이터를 활용함

  ex) 분류 : 카테고리에 따라 분류된 데이터를 학습 후 새로운 데이터가 들어왔을 때 카테고리 분류

  ex) 회귀 :불연속적인 데이터쌍을 바탕으로 연속된 패턴이나 경향, 그래프의 개형을 파악

- 비지도학습 (Un-supervised Learning)

  정답 라벨이 없는 데이터들을 비슷한 특징끼리 군집화

  ex) 클러스터링

- 강화학습 (Reinforcement Learning)

  ??

## 개발 환경 세팅

windows 10 pro

python 3.9.5

```powershell
python3 -m pip install numpy
python3 -m pip install pandas
python3 -m pip install matplotlib
python3 -m pip install seaborn
python3 -m pip install sklearn
```

~~4장, 9장에서 Microsoft Visual Studio Build Tools 필요~~

## Numpy

### import

```python
import numpy as np
```

numpy 대신 약어 np로 사용하는 것이 관례

### numpy.array()

```python
import numpy as np

arr1 =  np.array([1,2,3])
print(arr1)
```

```
[1 2 3]
```

python의 리스트 등을 ndarray로 변환

### ndarray.ndim ndarray.shape

```python
import numpy as np

arr =  np.array([1,2,3])
print(arr)
print(arr.ndim)
print(arr.shape)
print("----------------")
arr =  np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr)
print(arr.ndim)
print(arr.shape)
```

```
[1 2 3]
1
(3,)
----------------
[[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]]
3
(2, 2, 3)
```

np.ndim() : ndarray의 차원 return

np.shape() : ndarray의 차원과 크기를 tuple로 return

### ndarray.dtype ndarray.astype()

```python
import numpy as np

arr = np.array([1,2,3])
print(arr)
print(arr.dtype)
arr = arr.astype('bool')
print(arr)
print(arr.dtype)
arr = arr.astype('float64')
print(arr)
print(arr.dtype)
arr = arr.astype('str')
print(arr)
print(arr.dtype)

arr = np.array([1.1,2.2,3.3])
print(arr)
print(arr.dtype)
arr = arr.astype("int32")
print(arr)
print(arr.dtype)
```

```
[1 2 3]
int32
[ True  True  True]
bool
[1. 1. 1.]
float64
['1.0' '1.0' '1.0']
<U32
[1.1 2.2 3.3]
float64
[1 2 3]
int32
```

np.dtype() : ndarray 내의 데이터 타입 return

np.astype() : ndarray 내 데이터값의 타입 변환

### np.arange() np.zeros() np.ones()

```python
import numpy as np

arr = np.arange(5)
print(arr)
arr = np.arange(3,8)
print(arr)
arr = np.arange(0,9,2)
print(arr)
arr = np.arange(3,dtype='float64')
print(arr)

arr = np.zeros((5,))
print(arr)

arr = np.ones((2,2))
print(arr)
```

```
[0 1 2 3 4]
[3 4 5 6 7]
[0 2 4 6 8]
[0. 1. 2.]
[0. 0. 0. 0. 0.]
[[1. 1.]
 [1. 1.]]
```

np.arange() : 연속적인 값을 가진 ndarray return

np.zeros() : 0으로 초기화된 ndarray return

np.ones() : 1로 초기화된 ndarray return

### ndarray.reshape()

```python
import numpy as np

arr = np.array([[1,2,3],[4,5,6]])
print(arr)
print(arr.shape)
arr = arr.reshape((3,2))
print(arr)
print(arr.shape)
arr = arr.reshape((2,-1))
print(arr)
print(arr.shape)
```

```
[[1 2 3]
 [4 5 6]]
(2, 3)
[[1 2]
 [3 4]
 [5 6]]
(3, 2)
[[1 2 3]
 [4 5 6]]
(2, 3)
```

ndarray의 차원을 변환

인자로 -1 입력 시 호환될 수 있도록 자동 지정

단, 지정된 사이즈로 변경이 불가능하면 오류 발생

ex) (4,5) to (3,7) makes error

ex) (5,6) to (-1,8) makes error

### Indexing

#### Single Value

```python
import numpy as np

arr = np.arange(10)
print(arr)
print(arr[7])
print(arr[-3])
arr = np.array([[1,2,3],[4,5,6]])
print(arr)
print(arr[0][1])
```

```
[0 1 2 3 4 5 6 7 8 9]
7
7
[[1 2 3]
 [4 5 6]]
2
```

원하는 위치의 인덱스 값 지정하여 접근

인덱스 값을 음수로 지정할 경우 맨 뒤에서부터 순차적으로 접근

#### Slicing

```python
import numpy as np

arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(arr,'\n')
print(arr[:,:],'\n')
print(arr[:2,1:3],'\n')
print(arr[:,1])
```

```
[[0 1 2]
 [3 4 5]
 [6 7 8]] 

[[0 1 2]
 [3 4 5]
 [6 7 8]]

[[1 2]
 [4 5]]

[1 4 7]
```

Fancy Indexing과 달리 연속된 범위의 데이터 추출

#### Fancy Indexing

```python
import numpy as np

arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(arr,'\n')
print(arr[[0,2],:],'\n')
print(arr[2,[0,2]],'\n')
print(arr[:,[0,2]],'\n')
print(arr[[0,2]])
```

```
[[0 1 2]
 [3 4 5]
 [6 7 8]] 

[[0 1 2]
 [6 7 8]]

[6 8]

[[0 2]
 [3 5]
 [6 8]]

[[0 1 2]
 [6 7 8]]
```

Slicing과 달리 입력한 인덱스의 데이터만 추출

#### Boolean Indexing

```python
import numpy as np

arr = np.array([3,1,4,1,5,9,2])
print(arr)
print(arr<5)
print(arr[arr<5])
print(arr[[False, True, False, True, False, False, False]])
```

```
[3 1 4 1 5 9 2]
[ True  True  True  True False False  True]
[3 1 4 1 2]
[1 1]
```

### numpy.sort() ndarray.sort()

```python
import numpy as np

arr = np.array([3,1,4,1,5,9,2])
print(arr)
print(np.sort(arr))
print(arr)
print(arr.sort())
print(arr)
```

```
[3 1 4 1 5 9 2]
[1 1 2 3 4 5 9]
[3 1 4 1 5 9 2]
None
[1 1 2 3 4 5 9]
```

np.sort(ndarray)의 경우 원본을 유지하고 새로운 ndarray return

ndarray.sort()의 경우 return이 없으며 원본 자체를 sort

--------------------
sort 오름/내림차순 및 정렬행렬 인덱스 반환

내적 전치

pandas & dataframe 미완...
## 1. Numpy
### 단일 객체 저장 및 불러오기
```
array = np.arange(0, 10)

# .npy 파일에다가 저장하기
np.save("./save.npy", array)

# 불러오기
result = np.load('./save.npy')
```

### 복수 객체 저장 및 불러오기
```
# 복수 객체 저장을 위한 데이터 생성
array1 = np.arange(0, 10)
array2 = np.arange(0, 20)

# 저장하기
np.savez('./save.npz', array1=array1, array2=array2)

# 객체 불러오기
data = np.load('./save.npz')
result1 = data['array1']
result2 = data['array2']
```

### 원소 정렬
```
array_data.sort()
```

### 각 열 기준 정렬

```
array.sort(axis=0)

각 열을 기준으로 정렬 전 
 [[ 5  2  7  6]
 [ 2  3 10 15]]
각 열을 기준으로 정렬 후 
 [[ 2  2  7  6]
 [ 5  3 10 15]]
```


<hr>

## 2. OpenCV
```
# cv2는 버그가 많아서 이전 버전으로 실습 (버전은 계속 바뀜) 
pip install opencv-python==4.5.5.62
```

import numpy as np

# 복수 객체 저장을 위한 데이터 생성
array1 = np.arange(0, 10)
array2 = np.arange(0, 20)
print(array1, array2)

# 저장하기
np.savez('./save.npz', array1=array1, array2=array2)

# 객체 불러오기
data = np.load('./save.npz')
result1 = data['array1']
result2 = data['array2']

print("result1 >> ", result1)
print("result2 >> ", result2)
"""
result1 >>  [0 1 2 3 4 5 6 7 8 9]
result2 >>  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
"""

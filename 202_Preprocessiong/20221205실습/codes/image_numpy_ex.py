import numpy as np
import cv2

# 이미지 경로
x = cv2.imread("./cat.jpg", 0)  # 흑백 이미지
y = cv2.imread("./cat.jpg", 1)  # 컬러 이미지

# 여러개 파일 save .npz
np.savez("./image.npz", array1=x, array2=y)

# 압축 방법
np.savez_compressed("./image_compressed.npz", array1=x, array2=y)

# npz 데이터 로드
data = np.load("./image_compressed.npz")

result1 = data['array1']
result2 = data['array2']

cv2.imshow("result01", result1)
cv2.waitKey(0)

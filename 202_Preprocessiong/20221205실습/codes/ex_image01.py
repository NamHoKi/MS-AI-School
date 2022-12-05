import cv2

img_path = "./cat.jpg"
img = cv2.imread(img_path)

h, w, _ = img.shape

print("이미지 타입 : ", type(img))
print(f"이미지 높이 {h}, 이미지 넓이 {w}")
"""
이미지 타입 :  <class 'numpy.ndarray'>
이미지 높이 399, 이미지 넓이 600
"""

cv2.imshow("image show", img)
cv2.waitKey(0)

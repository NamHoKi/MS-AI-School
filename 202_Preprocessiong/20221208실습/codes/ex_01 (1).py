# 같은 크기의 이미지 블렌딩 실험
import cv2
import matplotlib.pyplot as plt
import numpy as np

large_img = cv2.imread('./ex_image.png')
watermakr = cv2.imread('./ex_image_logo.png')

print("large_image size >> ", large_img.shape)
print("watermakr image size >> ", watermakr.shape)

img1 = cv2.resize(large_img, (800, 600))
img2 = cv2.resize(watermakr, (800, 600))

print("img1 reize >>", img1.shape)
print("img2 reize >>", img2.shape)

"""
large_image size >>  (683, 1024, 3)
watermakr image size >>  (480, 640, 3)
img1 reize >> (600, 800, 3)
img2 reize >> (600, 800, 3)
"""
# 혼합 진행

# # 베이스 5:5
# blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

# # 9:1
# blended = cv2.addWeighted(img1, 9, img2, 1, 0)

# 1로 설정
blended = cv2.addWeighted(img1, 1, img2, 1, 0)
cv2.imshow("image show", blended)
cv2.waitKey(0)
# cv2.imshow("image large", img1)
# cv2.imshow("watermakr", img2)
# cv2.waitKey(0)

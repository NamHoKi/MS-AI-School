import cv2
import numpy as np
import matplotlib.pyplot as plt
img_gray = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY_INV)
# 3x3 Kenrnel
kernel = np.ones((3, 3), np.uint8)
print(kernel)
dilation = cv2.dilate(mask, kernel)

titles = ['image', 'mask', 'dilation']
images = [img_gray, mask, dilation]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

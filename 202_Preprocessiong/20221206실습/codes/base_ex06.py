import cv2
import numpy as np
from utils import image_show

img_path = "./car.jpg"
img = cv2.imread(img_path)

# 세피아 효과 필터
filter = np.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]])

sepia_img = cv2.transform(img, filter)
image_show(sepia_img)

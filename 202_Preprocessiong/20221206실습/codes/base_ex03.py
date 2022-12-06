import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.jpg')

# Creating out sharpening filter
filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

sharpen_img = cv2.filter2D(image, -1, filter)
cv2.imshow("org image ", image)
cv2.waitKey(0)
image_show(sharpen_img)

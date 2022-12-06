import cv2
import numpy as np
from utils import image_show

image = cv2.imread('./car.jpg')

filter = np.array([[3, -2, -3], [-4, 8, -6], [5, -1, -0]])
custom_image_filter = cv2.filter2D(image, -1, filter)

image_show(custom_image_filter)

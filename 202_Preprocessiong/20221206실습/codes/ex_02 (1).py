# 가우시안 블러
import cv2
from utils import image_show

image_path = "./cat.jpg"

# 이미지 읽기
image = cv2.imread(image_path)
print(image)

image_g_blury = cv2.GaussianBlur(image, (5, 5), 0)
image_show(image_g_blury)

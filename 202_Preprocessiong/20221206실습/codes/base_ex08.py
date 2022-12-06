import cv2
import matplotlib.pyplot as plt

# image loading and input image -> gray
img = cv2.imread("./Billiards.png", cv2.IMREAD_GRAYSCALE)

# 임계값 연산자의 출력을 마스크 라는 변수에 저장
# 230 보다 작으면 모든 값은 흰색 처리 / 230 보다 큰 모든 값은 검은색 이 됩니다.
_, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

titles = ['image', 'mask']
images = [img, mask]

for i in range(2):
    plt.subplot(1, 2, i+1),
    plt.imshow(images[i], 'gray'),
    plt.title(titles[i]),
    plt.xticks([]),
    plt.yticks([]),
plt.show()

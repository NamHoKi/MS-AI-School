import cv2

# 이미지 경로
image_path = "./cat.jpg"

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지 좌우 및 상하 반전
# 1 좌우 반전 0 상하 반전
dst_tmep1 = cv2.flip(image, 1)
dst_tmep2 = cv2.flip(image, 0)

cv2.imshow("dst_tmep1", dst_tmep1)
cv2.imshow("dst_tmep2", dst_tmep2)
cv2.waitKey(0)


# img90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # 시계 방향 으로 90도 회전
# img180 = cv2.rotate(image, cv2.ROTATE_180)  # 180도 회전
# img270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
# # 반시계 방향으로 90도 회전 = 시계방향 270도 회전
# print(image.shape)
# print(img90.shape)
# print(img270.shape)
# cv2.imshow("orginal image", image)
# cv2.imshow("rotate_90", img90)
# cv2.imshow("rotate_180", img180)
# cv2.imshow("rotate_270", img270)
# cv2.waitKey(0)

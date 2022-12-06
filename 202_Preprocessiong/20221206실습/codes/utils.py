import cv2


def image_show(image):
    cv2.imshow("show", image)
    cv2.waitKey(0)

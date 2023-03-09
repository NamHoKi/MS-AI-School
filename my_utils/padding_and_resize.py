# 이미지 주변에 padding을 넣고 300*300으로 변환, 저장하기.
import cv2
import numpy as np
import os
import glob


def padding_and_resize(file_path, save_path, result_size) :
    img = cv2.imread(file_path)
    if img is None :
        print(file_path, 'is None ...')
    else :

        # 이미지의 x, y가 result size 를 넘을 경우 작게해주기
        percent = 1
        if(img.shape[1] > img.shape[0]) :       # 이미지의 가로가 세보다 크면 가로를 300으로 맞추고 세로를 비율에 맞춰서
            percent = result_size/img.shape[1]
        else :
            percent = result_size/img.shape[0]

        img = cv2.resize(img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)

        # 이미지 범위 지정
        y,x,h,w = (0,0,img.shape[0], img.shape[1])

        # 그림 주변에 검은색으로 칠하기
        w_x = (result_size-(w-x))/2  # w_x = (300 - 그림)을 뺀 나머지 영역 크기 [ 그림나머지/2 [그림] 그림나머지/2 ]
        h_y = (result_size-(h-y))/2

        if(w_x < 0):         # 크기가 -면 0으로 지정.
            w_x = 0
        elif(h_y < 0):
            h_y = 0

        M = np.float32([[1,0,w_x], [0,1,h_y]])  #(2*3 이차원 행렬)
        img_re = cv2.warpAffine(img, M, (224, 224))

        # cv2.imshow("img_re", img_re)

        # 이미지 저장하기
        cv2.imwrite(save_path, img_re)

root_path = 'D:\\downloads\\dataset_filtering'
all_data_path = glob.glob(os.path.join(root_path, '*', '*', '*.png'))
for path in all_data_path :
    save_path = path.replace('downloads\\dataset_filtering', 'padding_and_resize224')
    p = save_path.split('\\')
    save_foler = os.path.join(p[0], p[1], p[2], p[3])
    os.makedirs(save_foler, exist_ok=True)
    padding_and_resize(path, save_path, 224)

import cv2
import glob
import os
import random
import numpy as np
import shutil

root_path = 'C:\\Users\\labadmin\\Downloads\\split_dataset\\03.split_dataset'
all_image_path = glob.glob(os.path.join(root_path, '*', 'images', '*.jpg'))

label_dict = {0:'1',
              1:'2',
              2:'3',
              3:'4',
              4:'5',
              5:'6',
              6:'7',
              7:'8',
              8:'9',
              9:'0',
              10:'m',
              11:'d'}


for image_path in all_image_path :
    # 1. image, label 매칭
    label_path = image_path.replace('\\images\\', '\\labels\\')[:-3] + 'txt'

    # 2. 작업 해놓은 yolo format 가져오기
    with open(label_path, 'r', encoding='utf-8') as f :
        lines = f.read().split('\n')


    # 3. 계산에 필요한 이미지 w, h 값 구하기
    origin_image = cv2.imread(image_path)
    image_h, image_w, _ = origin_image.shape

    # 4. 샤프닝 필터 정의
    sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpening_mask3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # 5. 샤프닝 이미지 생성
    s1 = cv2.filter2D(origin_image, -1, sharpening_mask1)
    s2 = cv2.filter2D(origin_image, -1, sharpening_mask2)
    s3 = cv2.filter2D(origin_image, -1, sharpening_mask3)
    s = [origin_image, s1, s2, s3]

    save_image_path = image_path.replace(root_path, 'D:\\0314_auto_box')
    save_image_path = save_image_path.replace('\\image\\', '\\images\\')[:-4] + '_$$_' + '.png'
    save_label_path = save_image_path.replace('\\images\\', '\\labels\\')[:-4] + '.txt'
    sp1 = save_image_path.split('\\')
    sp2 = save_label_path.split('\\')
    save_image_folder = os.path.join(sp1[0], sp1[1], sp1[2], sp1[3])
    save_label_folder = os.path.join(sp2[0], sp2[1], sp2[2], sp2[3])

    os.makedirs(save_image_folder, exist_ok=True)
    os.makedirs(save_label_folder, exist_ok=True)

    cnt = 0
    for i in range(len(s)) :
        sharpening_image = s[i]
        for j in range(0, 7, 2) :
            for k in range(0, 7, 2):
                save_image = save_image_path.replace('$$', str(cnt))
                save_label = save_label_path.replace('$$', str(cnt))
                cnt += 1
                for line in lines:
                    if line == '':
                        continue
                    label, x_c, y_c, w, h = line.split(' ')
                    label, x_c, y_c, w, h = int(label), float(x_c), float(y_c), float(w) * random.uniform(1.0 + (j*0.01), 1.01 + (j*0.01)), float(h) * random.uniform(1.0 + (k*0.01) , 1.01 + (k*0.01))

                    with open(save_label, 'a', encoding='utf-8') as f :
                        f.write(f'{label} {x_c} {y_c} {w} {h}\n')
                cv2.imwrite(save_image, sharpening_image)
        break
            # x1, y1, x2, y2 = int((x_c - (w * 0.5)) * image_w), int((y_c - (h * 0.5)) * image_h), int(
            #     (x_c + (w * 0.5)) * image_w), int((y_c + (h * 0.5)) * image_h)

            # try:
            #     cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #     cv2.putText(test_image, label_dict[label], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                 (0, 0, 255), 2)
            # except Exception as e:
            #     print(e)

    ## 가로로 합치기
    # result = cv2.hconcat([origin_image, test_image])

    ## 새로로 합치기
    # result2 = cv2.hconcat([s1, s2, s3])
    # cv2.imshow("result", result2)
    # cv2.waitKey(0)

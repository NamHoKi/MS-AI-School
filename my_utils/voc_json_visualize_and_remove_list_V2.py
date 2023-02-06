'''
<설정>
 1. 126, 127 번째 줄에 자신의 라벨 폴더경로, 이미지 폴더경로 설정
 2. 130번째 줄에 자기가 맡은 txt 파일 경로 설정
<사용법>
 1. s, esc를 제외한 아무 키를 누르면 이미지가 넘어감
 2. s를 누르면 보고 있는 이미지를 'remove_list.txt' 파일에 저장함 (문제 있는 이미지가 나오면 s 누르기, 저장되는 이미지명 확인, 콘솔창에 나옴)
 3. 콘솔창에 몇번째 이미지를 보고 있는지 (Count) , 현재 보고 있는 이미지의 경로 (Cur image), 출력되니 햇갈리거나 하면 확인
'''


import glob
import os
import json
import cv2
import copy


def read_json(json_path):
    # 본 데이터셋에서 학습에 필요한 정보만 읽어 반환
    with open(json_path, 'r', encoding="utf8") as j:
        json_data = json.load(j)

    images = json_data['image']
    annotations = json_data['annotations']

    filename = images['filename']
    height = images['resolution'][1]
    width = images['resolution'][0]

    annos = []
    for annotation in annotations:
        label = annotation['class']
        try:
            bbox = annotation['box']
        except Exception as e :
            # print('-' * 20, '박스없다')
            continue

        xmin, ymin = bbox[0], bbox[1]
        xmax, ymax = bbox[2], bbox[3]


        anno = {
                'label': label,
                'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]
        }
        annos.append(anno)

    data = {
        'filename': filename,
        'height': height,
        'width': width,
        'annos': annos
    }

    return data

def visualize(image_path, anno_data):
    # label과 bbox 시각화
    image = cv2.imread(image_path)

    for anno in anno_data['annos']:
        # rectangle
        pt1 = (anno['bbox'][0], anno['bbox'][1])
        pt2 = (anno['bbox'][2], anno['bbox'][3])
        image = cv2.rectangle(image, pt1, pt2, (250, 0, 250), 5)

        # text
        pt = (anno['bbox'][0], anno['bbox'][3] - 20)
        image = cv2.putText(image, anno['label'], pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 0, 250), 1, cv2.LINE_AA)

    # 원본 이미지가 크므로 원활한 시각화를 위해 이미지 크기 조절
    image = cv2.resize(image, (960, 540))

    return image


def resize(image_path, anno_data, size):
    # size: (new_width, new_height)
    # size 크기로 맞춰 이미지 resize 및 bbox 정보 수정

    width = anno_data['width']
    height = anno_data['height']

    image = cv2.imread(image_path)
    image = cv2.resize(image, (size[0], size[1]))

    width_ratio = size[0] / width
    height_ratio = size[1] / height

    new_data = copy.deepcopy(anno_data)
    for anno in new_data['annos']:
        xmin, ymin = anno['bbox'][0], anno['bbox'][1]
        xmax, ymax = anno['bbox'][2], anno['bbox'][3]

        xmin, xmax = xmin * width_ratio, xmax * width_ratio
        ymin, ymax = ymin * height_ratio, ymax * height_ratio

        anno['bbox'] = [int(xmin), int(ymin), int(xmax), int(ymax)]

    return image, new_data


def visualize_test(image, anno_data):
    # resize 결과 시각화 용 코드
    for anno in anno_data['annos']:
        # rectangle
        pt1 = (anno['bbox'][0], anno['bbox'][1])
        pt2 = (anno['bbox'][2], anno['bbox'][3])
        image = cv2.rectangle(image, pt1, pt2, (250, 0, 250), 5)

        # text
        pt = (anno['bbox'][0], anno['bbox'][3] - 20)
        image = cv2.putText(image, anno['label'], pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 3, cv2.LINE_AA)

    image = cv2.resize(image, (960, 540))

    return image


def main() :
    label_dict = {0: '01',
                  1: '02',
                  2: '03',
                  3: '04',
                  4: '05',
                  5: '06',
                  6: '07',
                  7: '08',
                  }

    cnt = 0


    # 경로 설정
    label_dir = 'D:\\0206\\g\\Training\\label\\1\\'
    image_dir = 'D:\\temp2\\'

    # json_paths = glob.glob(os.path.join(label_dir, '*.json'))
    with open('hk.txt', 'r', encoding='utf-8') as f :
        json_paths = f.read().split('\n')

    for json_path in json_paths :
        json_path = label_dir + json_path[:-4] + '.json'
        cnt += 1
        print('-'*50)
        print('Count      :', cnt)

        # 놓치거나 수정할 cnt 찾아가기
        if cnt < 0 :
            continue

        if not os.path.isfile(json_path):
            print('File does not exist:', json_path)
            pass

        anno_data = read_json(json_path)
        image_path = image_dir + json_path.split('\\')[-1][:-5] + '.jpg'

        image, anno_data = resize(image_path, anno_data, (1470, 810))
        image = visualize_test(image, anno_data)

        cv2.imshow('visual', image)

        print('Cur image  :', image_path)
        # 이미지 확인 & 키보드 클릭 이벤트 처리: keyboard 버튼 클릭과 (if문 하위에)동작 매칭 - 저장, 삭제 등
        while True:
            key = cv2.waitKey()
            if key == ord('s'):
                # txdft file save folder
                with open('./remove_list.txt', 'a') as f:
                    f.write(f'{image_path}\n')
                print('Save image :', image_path.split('\\')[-1])
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                exit()
            else:  # 위 if문에서 지정하지 않은 키보드 입력인 경우 다음 이미지로 넘어감
                break


if __name__ == '__main__':
    main()



# 2023년 1월 28일(토)
import torch.cuda
import os
import glob
import cv2

# device setting
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model call
model = torch.hub.load('ultralytics/yolov5', 'custom', path="../b64_yolov5n.pt")
# model = custom(path_or_model='.\\runs\\train\\exp3\\weights\\best.pt')
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.45 # NMS IoU threshold
model.to(DEVICE)

# image path list
# jpg파일을 가져옴
image_path_list = glob.glob(os.path.join("../dataset/test/images", "*.jpg"))

label_dict = {
        0: 'belt',
        1: 'no_belt',
        2: 'hoes',
        3: 'no_shoes',
        4: 'helmet',
        5: 'no_helmet'
    }

# cvt_label_dict = {v: k for k, v in label_dict.items()}

# 하나하나의 이미지 추출
for i in image_path_list:
    image_path = i
    # cv2 image read
    image = cv2.imread(image_path)



    # model input
    # 모델에 이미지를 넣어준다.
    output = model(image, size=640)
    # print(output.print())
    bbox_info = output.xyxy[0] # bounding box의 결과를 추출
    # for문을 들어가서 우리가 원하는 결과를 뽑는다.

    for bbox in bbox_info:
        # bbox에서 x1, y1, x2, y2, score, label_number의 결과를 가지고 온다.
        x1 = int(bbox[0].item())
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())

        score = bbox[4].item()
        label_number = int(bbox[5].item())
        # print(x1, y1, x2, y2, score, label_number) # 298 706 437 728 0.8252878785133362 7의 결과가 나오는 것을 볼 수 있다
        try :
            cv2.rectangle(image, (x1, y1), (x2, y2),(0, 255, 0), 2)
            cv2.putText(image, label_dict[label_number], (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, str(round(score, 4)), (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except Exception as e :
            print(e)

    cv2.imshow("test", image)
    cv2.waitKey(0)

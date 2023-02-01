import os
import glob
import json
import shutil


class Coco_json_to_yolo_txt() :
    def __init__(self, path, mode):
        self.all_image_path = glob.glob(os.path.join(path, mode, '*.jpg'))
        self.json_path = glob.glob(os.path.join(path, mode, '*.json'))
        self.mode = mode
        self.path = path
        self.label_dict = {}

        self.json2txt()


    def json2txt(self):
        # load json
        with open(self.json_path[0]) as f :
            json_object = json.load(f)

        # label
        for i, c in enumerate(json_object['categories']) :
            self.label_dict[i] = c['name']
        print(self.label_dict)
        exit()
        # images {'id': 0, 'license': 1, 'file_name': '113235_jpg.rf.b4273d3f8e92812554eb6b6536751721.jpg', 'height': 886, 'width': 767, 'date_captured': '2022-08-30T09:49:53+00:00'}
        images = json_object['images']

        # annotations {'id': 0, 'image_id': 0, 'category_id': 5, 'bbox': [237, 46, 299, 154], 'area': 46046, 'segmentation': [], 'iscrowd': 0}
        annotations = json_object["annotations"]

        for ann in annotations :
            image_id = ann['image_id']
            bbox = ann['bbox']  # x1 y1 w h
            label = ann['category_id']

            image = images[image_id]
            image_name = image['file_name']
            image_path = os.path.join(self.path, self.mode, image_name)

            h = image['height']
            w = image['width']

            yolo_x = round((int(bbox[0]) + (int(bbox[2]) // 2)) / w, 6)
            yolo_y = round((int(bbox[1]) + (int(bbox[3]) // 2)) / h, 6)
            yolo_w = round(int(bbox[2]) / w, 6)
            yolo_h = round(int(bbox[3]) / h, 6)

            # labels folder
            labels_folder = f'./dataset/{self.mode}/labels/'
            os.makedirs(labels_folder, exist_ok=True)

            file_name = image_name[:-4]
            with open(f'{labels_folder}/{file_name}.txt', 'a') as f:
                f.write(f'{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n')

            # images folder
            images_folder = f'./dataset/{self.mode}/images/'
            os.makedirs(images_folder, exist_ok=True)

            shutil.copy(image_path, images_folder + image_name)


test = Coco_json_to_yolo_txt('./dataset0130json', 'valid')

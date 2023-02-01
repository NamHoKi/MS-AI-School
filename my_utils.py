import os
import glob
from xml.etree.ElementTree import parse



# xml_dir = './wine labels_voc_dataset/'

# label_dict = {
#         'AlcoholPercentage': 0,
#         'Appellation AOC DOC AVARegion': 1,
#         'Appellation QualityLevel': 2,
#         'CountryCountry': 3,
#         'Distinct Logo': 4,
#         'Established YearYear': 5,
#         'Maker-Name': 6,
#         'Organic, Sustainable': 7,
#         'Sweetness-Brut-SecSweetness-Brut-Sec': 8,
#         'TypeWine Type': 9,
#         'VintageYea': 10
#     }

# class Voc_to_yolo_convter() :
#     def __init__(self, xml_paths):
#         self.xml_path_list = glob.glob(os.path.join(xml_paths, '*.xml'))


#     def voc_xyxy_show(self, x1, y1, x2, y2, label_name, file_name):
#         image_path = os.path.join('./wine labels_voc_dataset', file_name)
#         image = cv2.imread(image_path)
#         img_rect = cv2.rectangle(image, (x1, y1), (x2, y1), (0, 255, 0), 2)

#         return img_rect


#     def get_voc_to_yolo(self):
#         for xml_path in self.xml_path_list :
#             tree = parse(xml_path)
#             root = tree.getroot()

#             # get file name
#             file_name = root.findall('filename').text

#             # get image size
#             size_meta = root.findall('size')
#             img_width = int(size_meta[0].find('width').text)
#             img_height = int(size_meta[0].find('height').text)
#             print(img_width, img_height)

#             # object meta
#             object_metas = root.findall('object')

#             # box info get
#             for object_meta in object_metas :
#                 # label_name
#                 object_label = object_meta.find('name').text

#                 # bbox
#                 xmin = object_meta.find('bndbox').findtext('xmin')
#                 xmax = object_meta.find('bndbox').findtext('xmax')
#                 ymin = object_meta.find('bndbox').findtext('ymin')
#                 ymax = object_meta.find('bndbox').findtext('ymax')

#                 # print(object_label, xmin, ymin, xmax, ymax)

#                 # voc to yolo
#                 yolo_x = round(((int(xmin) + int(xmax)) / 2) / img_width, 6)
#                 yolo_y = round(((int(ymin) + int(ymax)) / 2) / img_height, 6)
#                 yolo_w = round((int(xmax) - int(xmin))/img_width, 6)
#                 yolo_h = round((int(ymax) - int(ymin))/img_height, 6)

#                 # print(yolo_x, yolo_y, yolo_w, yolo_h)
                
                
def voc_xml_to_yolov5txt(xml_path, txt_folder, file_name) :
    '''
    voc xml -> yolo5 txt
    :param path: .xml file path , save txt foloder
    :return: None
    '''

    label_dict = {
        'AlcoholPercentage': 0,
        'Appellation AOC DOC AVARegion': 1,
        'Appellation QualityLevel': 2,
        'CountryCountry': 3,
        'Distinct Logo': 4,
        'Established YearYear': 5,
        'Maker-Name': 6,
        'Organic, Sustainable': 7,
        'Sweetness-Brut-SecSweetness-Brut-Sec': 8,
        'TypeWine Type': 9,
        'VintageYea': 10
    }

    tree = parse(xml_path)
    root = tree.getroot()
    w = int(root.find('size').find('width').text)
    h = int(root.find('size').find('height').text)
    img_metas = root.findall('object')

    for img_meta in img_metas:
        try:
            for member in root.findall('object'):
                # label
                label = label_dict[member.find('name').text]
                # xmin = left corner x-coordinates
                xmin = int(member.find('bndbox').find('xmin').text)
                # xmax = right corner x-coordinates
                xmax = int(member.find('bndbox').find('xmax').text)
                # ymin
                ymin = int(member.find('bndbox').find('ymin').text)
                # ymax
                ymax = int(member.find('bndbox').find('ymax').text)

                # txt file save folder
                os.makedirs(txt_folder, exist_ok=True)

                yolo_x = round((int(xmin) + ((xmax - xmin) // 2)) / w, 6)
                yolo_y = round((int(ymin) + ((int(ymax) - int(ymin)) // 2)) / h, 6)
                yolo_w = round((int(xmax) - int(xmin)) / w, 6)
                yolo_h = round((int(ymax) - int(ymin)) / h, 6)

                # txt save
                with open(f'{txt_folder}/{file_name}.txt', 'a') as f:
                    f.write(f'{label} {yolo_x} {yolo_y} {yolo_w} {yolo_h} \n')

        except Exception as e:
            pass


# voc_xml_to_txt('./1edcbf2b-3db4-4542-8253-a8b1a1f42de4-1_jpg.rf.1a73ec6589734b3c1628cee164cedd9f.xml', '../voc_xml_to_yolo_txt')

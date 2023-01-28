import os
import glob
from xml.etree.ElementTree import parse

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

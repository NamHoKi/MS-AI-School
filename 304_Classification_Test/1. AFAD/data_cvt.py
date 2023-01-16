import os
import glob
from PIL import Image


all_img_path = glob.glob(os.path.join('./dataset/','*', '*', '*.png'))

for img_path in all_img_path :
    img = Image.open(img_path).convert('RGB') # jpg - > png
    img.save(img_path)
    print(img_path)

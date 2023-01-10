import glob
import os
from PIL import Image


def image_synthesis(label):
    img_path=f"./del_back/{label}"
    img_list = glob.glob(os.path.join(img_path,"*.png"))
    back_list = glob.glob(os.path.join("./utils/background/","*.jpeg"))    
    cnt = 0
    for i in img_list:
        for j in back_list:
            
            my_image = Image.open(j)
            my_image = my_image.resize((224, 224))
            watermark = Image.open(i)
            watermark = watermark.resize((224, 224))    # 배경제거된 상품 이미지 사이즈 결정
            x = my_image.size[0] - watermark.size[0]    # 새로운 배경에 넣을 좌표 설정 부분
            y = my_image.size[1] - watermark.size[1]
            my_image.paste(watermark, (x,y), watermark) # 배경에 이미지 합성
            my_image.save(f'./test/{label}/{label}_{cnt}.png')

            cnt += 1
img_path = "./del_back/"
label_list = ['jinro', 'choco']
for i in label_list:
    image_synthesis(i)


import os
from PIL import Image


os.chdir('./0106/seven/test/') #해당 폴더로 이동
files = os.listdir(os.getcwd()) #현재 폴더에 있는 모든 파일을 list로 불러오기
cnt = 0
for file in files:

   
    img = Image.open(file) #이미지 불러오기
    img_size = img.size #이미지의 크기 측정
        #직사각형의 이미지가 256x512 이라면, img_size = (256,512)가 된다.
    x = img_size[0] #넓이값
    y = img_size[1] #높이값
   
    if x != y:
        size = max(x, y)    
        resized_img = Image.new(mode = 'RGB', size = (size, size), color = (0, 0, 0))
        offset = (round((abs(x - size)) / 2), round((abs(y - size)) / 2))
        resized_img.paste(img, offset)
        resized_img = resized_img.resize((224, 224))
        resized_img.save('padding' + str(cnt) + '.png')
        cnt += 1

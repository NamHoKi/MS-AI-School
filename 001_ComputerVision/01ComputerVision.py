'''
Computer Vision Object Dectection
Computer Vision API를 사용해서 이미지속에 있는 사물을 인식하는 데모 입니다.
네트워크 통신을 위해서 requests 패키지를 import 합니다.
'''

from email import header
from symbol import subscript
import requests

'''
이미지처리를 위해서 matplotlib, pyplot, Image, BytesIO 세 개의  패키지를 import 합니다.
matplotlib, pyplot는 import 할 때 시간이 조금 걸릴 수 있습니다.
'''

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import json

'''
Subscription Key와 접속에 필요한 URL을 설정합니다.
'''

subscription_key = '키1 값 입력' #key 1 value
vision_base_url = '엔드포인트/vision/v2.0/' #엔드포인트 + viosn/v2.0

analyze_url = vision_base_url + 'analyze' # 이미지 분석을 위한 주소

'''
분석에 사용되는 이미지를 확인 합니다.
https://img3.yna.co.kr/photo/yna/YH/2019/12/31/PYH2019123113530001300_P4.jpg
'''

image_url = 'https://img3.yna.co.kr/photo/yna/YH/2019/12/31/PYH2019123113530001300_P4.jpg'

con = requests.get(image_url).content #requests.get(image_url) -> <Respones [200]> -> .content -> image 바이너리
byte = BytesIO(con)
image = Image.open(byte)

# -> image = Image.open(BytesIO(requests.get(image_url).content))

# image.show()

headers = {'Ocp-Apim-Subscription-key':subscription_key} # Ocp-Apim-Subscription-key - azuer에서 정해놓은 값
params  = {'visualFeatures':'Categories,Description,Color'} # https://learn.microsoft.com/ko-kr/azure/cognitive-services/computer-vision/how-to/call-analyze-image?tabs=rest
data = {'url' : image_url}

response = requests.post(analyze_url, headers = headers, params = params, json = data) # web 호출 방법 : get or post

result = response.json()

print(result)

image_caption = result['description']['captions'][0]['text']

print(image_caption)


'''
Object Detection
'''

objectDetection_url = vision_base_url + 'detect'

image_url = 'https://mblogthumb-phinf.pstatic.net/MjAyMDA5MDdfMjQ1/MDAxNTk5NDY1MjUxMjM4.zbBfDyquP67Utlw2d6pFOtHqnJyfkukH3PTDgDTg8Zkg.qQWiX02sgIaExMrU-guWXKDRsmnGBBxeS_bz2Ioy8YUg.PNG.vet6390/%EA%B0%95%EC%95%84%EC%A7%80_%EA%B3%A0%EC%96%91%EC%9D%B4_%ED%95%A8%EA%BB%98_%ED%82%A4%EC%9A%B0%EA%B8%B0.PNG?type=w800'

image = Image.open(BytesIO(requests.get(image_url).content))

# image.show()

headers = {'Ocp-Apim-Subscription-key':subscription_key} # Ocp-Apim-Subscription-key - azuer에서 정해놓은 값
params  = {'visualFeatures':'Categories,Description,Color'} # https://learn.microsoft.com/ko-kr/azure/cognitive-services/computer-vision/how-to/call-analyze-image?tabs=rest
data = {'url' : image_url}

reponse = requests.post(objectDetection_url, headers = headers, params = params, json = data)

result = reponse.json()

print(result)


# 시각화

from PIL import Image, ImageDraw, ImageFont

draw = ImageDraw.Draw(image)

# boundingBox를 위한 함수
def DrawBox(detectData):
    objects = detectData['objects']

    for obj in objects:
        print(obj)

        rect = obj['rectangle']

        x = rect['x']
        y = rect['y']
        w = rect['w']
        h = rect['h']

        draw.rectangle(((x,y),(x+w,y+h)),outline='red')

        objectName = obj['object']
        draw.text((x,y),text=objectName,fill='red')


DrawBox(result)

image.show()

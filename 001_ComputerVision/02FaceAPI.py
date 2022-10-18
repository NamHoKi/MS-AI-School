from symbol import subscript
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

subscription_key = 'key 1 입력' # key 1 value
face_api_url = '엔드포인트 + face/v1.0/detect 입력' # 엔드포인트 + face/v1.0/detect

image_url = 'http://photo.sentv.co.kr/photo/2021/08/24/20210824093751.jpg'

image = Image.open(BytesIO(requests.get(image_url).content))
# image.show() # 이미지 불러오기 테스트

headers = {'Ocp-Apim-Subscription-Key': subscription_key}

params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'Smile'
}

data = {'url': image_url}

response = requests.post(face_api_url, params=params, headers=headers,json=data)
faces = response.json()

print(faces)

draw = ImageDraw.Draw(image)

def DrawBox(faces):

  for face in faces:
    rect = face['faceRectangle']
    left = rect['left']
    top = rect['top']
    width = rect['width']
    height = rect['height']

    draw.rectangle(((left,top),(left+width,top+height)),outline='red')

    face_attributes = face['faceAttributes']
    smile = face_attributes['smile']
    draw.text((left,top),str(smile),fill='red')

DrawBox(faces)
image.show()

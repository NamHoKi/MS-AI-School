import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

subscription_key = '601478c2f16e4ee89eebf0f8028dca96'
vision_base_url = 'https://labuser42computervosion.cognitiveservices.azure.com/vision/v2.0/'
ocr_url = vision_base_url + 'ocr'

# image_url = 'https://i.stack.imgur.com/WiDpa.jpg'
image_url = "https://www.unikorea.go.kr/unikorea/common/images/content/peace.png"

image = Image.open(BytesIO(requests.get(image_url).content))

# image.show()

headers = {'Ocp-Apim-Subscription-Key': subscription_key}
params = {'language': 'unk', 'detectOrientation': 'true'}
data = {'url': image_url}

response = requests.post(ocr_url, headers=headers, params=params, json=data)

result = response.json()

print(result)

for region in result['regions']:
    lines = region['lines']

    for line in lines:
        words = line['words']

        for word in words:
            text = word['text']
            print(text)
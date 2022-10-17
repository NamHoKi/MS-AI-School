<h3>
Computer Vision Object Dectection<br>
Computer Vision API를 사용해서 이미지속에 있는 사물을 인식하는 데모 입니다.<br>
네트워크 통신을 위해서 requests 패키지를 import 합니다.
</h3>

```
from email import header
from symbol import subscript
import requests
```

<p>이미지처리를 위해서 matplotlib, pyplot, Image, BytesIO 세 개의  패키지를 import 합니다.</p>
<p>matplotlib, pyplot는 import 할 때 시간이 조금 걸릴 수 있습니다.</p>

```
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import json
```


<p>Subscription Key와 접속에 필요한 URL을 설정합니다.</P>

```
subscription_key = '601478c2f16e4ee89eebf0f8028dca96' #key 1 value
vision_base_url = 'https://labuser42computervosion.cognitiveservices.azure.com/vision/v2.0/' #엔드포인트 + viosn/v2.0

analyze_url = vision_base_url + 'analyze' # 이미지 분석을 위한 주소
```

<p>분석에 사용되는 이미지를 확인 합니다.<p>

[이미지 링크](https://img3.yna.co.kr/photo/yna/YH/2019/12/31/PYH2019123113530001300_P4.jpg)

```
image_url = 'https://img3.yna.co.kr/photo/yna/YH/2019/12/31/PYH2019123113530001300_P4.jpg'

con = requests.get(image_url).content
# requests.get(image_url)  : <Respones [200]>
# .content                 : image binary
byte = BytesIO(con)
image = Image.open(byte)
```
<p>위의 마지막 3줄의 코드를 한줄로 줄일 수 있다.</p>


```
image = Image.open(BytesIO(requests.get(image_url).content))
```
```
image.show() # 이미지 불러오는지 확인
```

[설명서 링크](https://learn.microsoft.com/ko-kr/azure/cognitive-services/computer-vision/how-to/call-analyze-image?tabs=rest)
```
headers = {'Ocp-Apim-Subscription-key':subscription_key}
# Ocp-Apim-Subscription-key - azuer에서 정해놓은 값
params  = {'visualFeatures':'Categories,Description,Color'}
data = {'url' : image_url}

response = requests.post(analyze_url, headers = headers, params = params, json = data) # web 호출 방법 : get or post

result = response.json()

print(result)
```
```
[출력]
{'categories': [{'name': 'people_crowd', 'score': 0.99609375}], 'color': {'dominantColorForeground': 'Black', 'dominantColorBackground': 'Grey', 'dominantColors': ['Black', 'Grey'], 'accentColor': '927A39', 'isBwImg': False, 'isBWImg': False}, 'description': {'tags': ['person', 'crowd', 'building', 'street', 'large', 'people', 'standing', 'woman', 'man', 'walking', 'city', 'group', 'many', 'holding', 'sign', 'traffic', 'playing', 'air', 'crowded'], 'captions': [{'text': 'a group of people standing in front of a crowd', 'confidence': 0.861677475146408}]}, 'requestId': 'ecfda47a-d948-4886-921b-cad7df3465b5', 'metadata': {'height': 675, 'width': 1024, 'format': 'Jpeg'}}
```
<p>json 타입으로 result에 저장</p>

```
image_caption = result['description']['captions'][0]['text']

print(image_caption)
```
```
[출력]
a group of people standing in front of a crowd
```



<h3>Object Detection</h3>

```
objectDetection_url = vision_base_url + 'detect'

image_url = 'https://mblogthumb-phinf.pstatic.net/MjAyMDA5MDdfMjQ1/MDAxNTk5NDY1MjUxMjM4.zbBfDyquP67Utlw2d6pFOtHqnJyfkukH3PTDgDTg8Zkg.qQWiX02sgIaExMrU-guWXKDRsmnGBBxeS_bz2Ioy8YUg.PNG.vet6390/%EA%B0%95%EC%95%84%EC%A7%80_%EA%B3%A0%EC%96%91%EC%9D%B4_%ED%95%A8%EA%BB%98_%ED%82%A4%EC%9A%B0%EA%B8%B0.PNG?type=w800'

image = Image.open(BytesIO(requests.get(image_url).content))


headers = {'Ocp-Apim-Subscription-key':subscription_key} # Ocp-Apim-Subscription-key - azuer에서 정해놓은 값
params  = {'visualFeatures':'Categories,Description,Color'} # https://learn.microsoft.com/ko-kr/azure/cognitive-services/computer-vision/how-to/call-analyze-image?tabs=rest
data = {'url' : image_url}

reponse = requests.post(objectDetection_url, headers = headers, params = params, json = data)

result = reponse.json()

print(result)
```
```
[출력]
{'objects': [{'rectangle': {'x': 211, 'y': 35, 'w': 349, 'h': 407}, 'object': 'dog', 'confidence': 0.543, 'parent': {'object': 'mammal', 'confidence': 0.929, 'parent': {'object': 'animal', 'confidence': 0.949}}}, {'rectangle': {'x': 8, 'y': 125, 'w': 237, 'h': 347}, 'object': 'cat', 'confidence': 0.824, 'parent': {'object': 'mammal', 'confidence': 0.89, 'parent': {'object': 'animal', 'confidence': 0.891}}}], 'requestId': 'a7488a16-6b7f-4a78-a375-73ccc2e036e3', 'metadata': {'height': 482, 'width': 651, 'format': 'Png'}}
{'rectangle': {'x': 211, 'y': 35, 'w': 349, 'h': 407}, 'object': 'dog', 'confidence': 0.543, 'parent': {'object': 'mammal', 'confidence': 0.929, 'parent': {'object': 'animal', 'confidence': 0.949}}}
{'rectangle': {'x': 8, 'y': 125, 'w': 237, 'h': 347}, 'object': 'cat', 'confidence': 0.824, 'parent': {'object': 'mammal', 'confidence': 0.89, 'parent': {'object': 'animal', 'confidence': 0.891}}}
```

# 시각화

```
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

        #사각형 그리기
        draw.rectangle(((x,y),(x+w,y+h)),outline='red')

        #이름 태그
        objectName = obj['object']
        draw.text((x,y),text=objectName,fill='red')


DrawBox(result)

image.show()
```

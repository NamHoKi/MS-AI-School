# [MS Azure](https://portal.azure.com/)

<hr>

  ## I. Start
    [리소스 생성]
    리소스 그룹 만들기 -> 리소스 만들기 -> 원하는 항목 검색 -> 만들기 (알맞은 값 설정)
    
    [리소스 사용]
    1. 코드로 사용 or 웹에서 실행
    2. 리소스 관리 -> 키 및 엔드포인트로 연동 -> 사용


<hr>

  ## II. Cognitive Services
  ### 1. Computer Vision
  #### 객체 검출
    1. MS Azure에 만가지가 넘는 객체가 이미 학습되어 있음
    2. 어떤 객체를 검출할지 target 설정

![catdog1](https://user-images.githubusercontent.com/48282708/196424673-7fb6210c-35a0-471f-b9e9-4a13b2677c6d.png)

    

<hr>

  ### 2. Face API
  #### 얼굴 인식
![face1](https://user-images.githubusercontent.com/48282708/196424453-7deb61b7-f6f4-43e8-bb0b-5f0053f2fb2f.png)

<hr>

  ### 3. Custom Vision
  ####  

<hr>

  ### 4. OCR
  ### 글자 인식
  #### 인식할 이미지 : [이미지 링크](https://www.unikorea.go.kr/unikorea/common/images/content/peace.png)
  ```
  [출력]
  {'language': 'ko', 'textAngle': 0.0, 'orientation': 'Up', 'regions': [{'boundingBox': '45,125,95,36', 'lines': [{'boundingBox': '45,125,95,17', 'words': [{'boundingBox': '45,125,46,17', 'text': '평화와'}, {'boundingBox': '95,125,45,17', 'text': '번영의'}]}, {'boundingBox': '70,144,46,17', 'words': [{'boundingBox': '70,144,46,17', 'text': '한반도'}]}]}]}
  평화와
  번영의
  한반도
  ```

<hr>
  
  
  
  
  ### 5. Chat bot

# Object detection

<hr>

라벨링 사이트 : https://www.cvat.ai/
MMDetection Tutorial : https://mmdetection.readthedocs.io/en/latest/tutorials/

<hr>
# voc

<folder> : 데이터셋 디렉토리명

<filename> : 파일이름

<source> : 이미지의 출처가 어디냐를 말하는듯

<owner> : 이 이미지의 소유자를 말하는듯(from flickr)

<size> : 이미지의 width, height, channel에 대한 정보

<segmented> : 이미지가 segmentation에 사용될 수 있도록 label되었는지 유무(ex, 000032.jpg)

<object> : 사진 속 object에 대한 정보

     <name> : object의 클래스 이름

     <pose> : object가 바라보고 있는 방향

     <truncated> : 이미지 상에 object가 잘려 일부만 나오는지

     <difficult> : 인식하기 어려운지 -> 보통 object 크기가 작으면 difficult가 1로 설정되는 듯 하다.(ex, 000862.jpg)

     <bndbox> : 이 object에 대한 box 정보

          <xmin> : box의 왼쪽 상단 x축 좌표값

          <ymin> : box의 왼쪽 상단 y축 좌표값

          <xmax> : box의 오른쪽 하단 x축 좌표값

          <ymax> : box의 오른쪽 하단 y축 좌표값

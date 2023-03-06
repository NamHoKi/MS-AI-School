# I. Classification 

* [PyTorch and Albumentations for image classification](https://albumentations.ai/docs/examples/pytorch_classification/)
* https://pytorch.org/get-started/previous-versions/

```
set env
$ conda create -n AI python=3.8
$ conda activate AI
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
$ 
$ pip install opencv-python==4.5.4.60
```


## 1. 데이터 분포 확인 



## 2. 데이터 나눈 작업 

    * train , val , test -> 8 : 1 : 1 # 학습데이터가 충분한 경우

    * 예외 : train, val -> 9 : 1 #학습데이터가 부족한경우 

    * image 저장 포멧 .png 

* 각 라벨별 만장이 필요 !! 최소한 !! 서비스 목적으로 한다면 ! 

-> 어려워요 !! 왜 ? 구하기 힘들어요 ! 

-> 단 : 데이터가 많다고 좋은건 아닙니다 !! 

-> 다 비슷해요 !! >> 오버피팅 (다양성이 필요하다)

나이 성별 계절(봄, 여름, 가을, 겨울) 장소(???) 시간 (아침, 점심, 저녁, 새벽), 날씨(비, 눈, 등등)



## 3. Custom dataset 구축 !! -> pytorch dataset 상속 -> class 

* __init__

이미지 폴더에서 이미지 경로 가져오기 -> list 

transform -> 정의 !! 

* __getitem__

이미지 경로가 담겨있는 list -> 인덱스 추출 -> for

이미지 풀 경로 -> cv2 PIL 이미지 open 

라벨 필요합니다. 

   -> 딕셔너리, if -> 폴더명 기준 ! 

   -> image.png  -> 이미지 파일 이름 기준 ! 

   -> label.txt , label.csv -> filename 해당하는 라벨 지정되어있습니다. 

   -> 폴더명 , 이미지명 기준 

어그멘테이션 적용 !! 

  return 이미지, 라벨 

* __len__

   전체 데이터 길이 반환-> list -> len()


 

## 4. 학습에 필요한 코드 

* device
```
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* transform
1. train_transfrom
2. val_transform
3. test_transform

```
# 1. train_transform 예시 (albumentation)

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.SmallestMaxSize(max_size = 256),
    A.Resize(height=224, width=224),
    
    ⋯

    A.RandomShadow(p=0.5),
    A.RandomFog(p=0.4),
    A.RandomSnow(p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                       rotate_limit=15, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std =(0.229,0.224,0.225)),
    ToTensorV2()
])

# 2. val_transform
val_transform = A.Compose([

   ⋯
   
   ])
   
# 3. test_transform = ⋯

```

* dataset

* DataLoader

   * [DataLoader Parameter](https://subinium.github.io/pytorch-dataloader/)
 

## 5. 테스트

test_transforms = A.compose

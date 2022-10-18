#%%
print('11111111111111')
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

ENDPOINT_Training = '트레이닝 엔드포인트 입력'
ENDPOINT_Prediction = '예측 엔드포인트 입력'

training_key = '트레이닝 키 값 입력'
prediction_key = '예측 키 값 입력'
prediction_resource_id = '예측 리소스 아이디 입력'

# publish_iteration_name = 'classfyModel'
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT_Training, credentials) # https://github.com/Azure-Samples/cognitive-services-quickstart-code/blob/master/python/CustomVision/ImageClassification/CustomVisionQuickstart.py

print("Creating project...")
project = trainer.create_project("Labuser42 Project")

Jjajangmyeon_tag = trainer.create_tag(project.id, "Jjajangmyeon")
Champon_tag = trainer.create_tag(project.id, "Champon")
Tangsuyug_tag = trainer.create_tag(project.id, "Tangsuyug")

#%%
# 여기서 태그에 이미지 추가를 해줘야함
# 실습에선 수동으로 이미지 추가

print('Training....')
iteration = trainer.train_project(project.id)
while (iteration.status != 'Completed'):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print('Training status' + iteration.status)

    time.sleep(2)

print('Done!')

#%%
# prediction

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient


prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT_Prediction, prediction_credentials)

target_image_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSiCfYeFRBUmW6gvlU2JvsQCcsrOFcJkBdDeA&usqp=CAU'
result = predictor.classify_image_url(project.id, "greatwall42", target_image_url)

#%%

for prediction in result.predictions:
    print('\t' + prediction.tag_name,": {0:.2f}%".format(prediction.probability * 100))
# %%

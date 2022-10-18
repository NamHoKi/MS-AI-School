from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

ENDPOINT = 'https://labuser42.cognitiveservices.azure.com/'

training_key = 'training key 입력' # training key
prediction_key = 'prediction key 입력' # prediction key
prediction_resource_id = 'prediction resource id 입력' # prediction resource id

publish_iteration_name = 'classfyModel'
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
# https://github.com/Azure-Samples/cognitive-services-quickstart-code/blob/master/python/CustomVision/ImageClassification/CustomVisionQuickstart.py

print("Creating project...")
project = trainer.create_project("Labuser42 Project")

Jjajangmyeon_tag = trainer.create_tag(project.id, "Jjajangmyeon")
Champon_tag = trainer.create_tag(project.id, "Champon")
Tangsuyug_tag = trainer.create_tag(project.id, "Tangsuyug")


# 여기서 태그에 이미지 추가를 해줘야함
# 실습에선 수동으로 이미지 추가

print('Training....')
iteration = trainer.train_project(project.id)
while (iteration.status != 'Completed'):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print('Training status' + iteration.status)

    time.sleep(2)

    print('Done!')


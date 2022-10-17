from symbol import subscript
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

subscript_key = 'ff75fdf6d10f493e82457c7278b31bcb' # key 1 value
face_api_url = 'https://labuser42face.cognitiveservices.azure.com/face/v1.0/detect' # 엔드포인트 + face/v1.0/detect


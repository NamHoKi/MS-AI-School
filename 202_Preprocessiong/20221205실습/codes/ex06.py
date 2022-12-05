import pytagcloud
import webbrowser
# pip install pytagcloud
# pip uninstall pytagcloud -> 삭제 uninstall
# pip install pygame
# pip install simplejson

tag = [('cat', 120), ('dog', 50), ('python', 110), ('java', 70), ('DB', 10)]
# 테그화
tag_list = pytagcloud.make_tags(tag, maxsize=50)
pytagcloud.create_tag_image(
    tag_list, "word_cloud.jpg", size=(900, 600), rectangular=False)
webbrowser.open('word_cloud.jpg')

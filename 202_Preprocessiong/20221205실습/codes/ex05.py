# 단어 중요도에 가중치 부여하기
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = np.array(
    ['I love Brazil. Brazil!', 'Sweden is best', 'Germany beats both'])

# tf-idf 특성 행렬을 만듭니다.
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
feature_matrix.toarray()  # tf-idf 특성 행렬을 밀집 배열로 확인
# print(feature_matrix)
# 특성 이름 을 확인
tf = tfidf.vocabulary_
print("...", tf)

"""
{'love': 6, 'brazil': 3, 'sweden': 7, 'is': 5, 'best': 1, 'germany': 4, 'beats': 0, 'both': 2}
"""

# 불용어 삭제
from nltk.corpus import stopwords
import nltk

# 불용어 데이터를 다운로드 -> 179개 입니다.
nltk.download('stopwords')

# 단어 토큰을 만듭니다.
tokenized_words = ['i', 'am', 'the', 'of', 'to', 'go', 'store', 'and', 'park']

# 불용어 로드
stop_words = stopwords.words('english')
print("불용어 리스트 길이 >> ", len(stop_words))
print("불용어 리스트 >> ", stop_words)

# 불용어 삭제
for word in tokenized_words:
    if word not in stop_words:
        print(word)
# [word for word in tokenized_words if word not in stop_words]

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
# pip install nltk

# 구두점 데이터를 다운로드 합니다.
nltk.download('punkt')

# 텍스트 데이터 생성 합니다
string_temp = "The science of today is the technology of tomorrow"
# 단어를 토큰으로 나눕니다.
token_temp = word_tokenize(string_temp)
print(token_temp)

string_temp01 = "The science of today is the technology of tomorro. Tomorrow is today. Is day."
sent_data = sent_tokenize(string_temp01)
print(sent_data)

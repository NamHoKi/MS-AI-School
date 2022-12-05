from nltk.stem.porter import PorterStemmer

# 단어 토큰을 만들기
tokneized_words = ['i', 'am',  'humbled',
                   'by', 'this', 'traditional', 'meeting']

# 어간 추출기를 만들기
porter = PorterStemmer()

word_list = []
# 어간 추출기를 적용합니다.
for word in tokneized_words:
    word_list.append(porter.stem(word))

print(word_list)

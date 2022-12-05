# 넷플릭스 영화 추천 시스템 구현
import pandas as pd
import numpy as np
from math import sqrt

# 데이터 읽어오기
# 1205 현재 위치 -> 2개 csv 존재, ex07.py
movies = pd.read_csv("./movies.csv")
ratings = pd.read_csv("./ratings.csv")

# 아이템 기반 협업 필터링 구현
data = pd.merge(ratings, movies, on="movieId")
column = ['userId', 'movieId', 'rating', 'title', 'genres']
data = data[column]

moviedata = data.pivot_table(index="movieId", columns='userId')['rating']

# NaN 값을 -1 로 변경
moviedata.fillna(-1, inplace=True)


def sim_distance(data, n1, n2):
    # kdd 유사도 함수
    sum = 0
    # 두 사용자가 모두 본 영화를 기준
    for i in data.loc[n1, data.loc[n1, :] >= 0].index:
        if data.loc[n2, i] >= 0:
            # 누적합
            sum += pow(data.loc[n1, i]-data.loc[n2, i], 2)
    return sqrt(1/(sum+1))  # 유사도 형식으로 출력


def top_match(data, name, rank=5, simf=sim_distance):
    # 나와 유사도 가 높은 유저 매칭
    simList = []
    for i in data.index[-10:]:
        if name != i:
            simList.append((simf(data, name, i), i))
        simList.sort()
        simList.reverse()
        return simList[:rank]

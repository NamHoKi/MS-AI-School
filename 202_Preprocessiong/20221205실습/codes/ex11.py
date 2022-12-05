import pandas as pd

dataframe = pd.DataFrame()
temp = dataframe['date'] = pd.date_range('1/1/2022', periods=150, freq='w')
# 년 월 일 시 분에 대한 특성을 만듭니다.
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

print(dataframe.head(30))

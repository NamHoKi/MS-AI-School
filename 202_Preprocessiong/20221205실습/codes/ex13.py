import pandas as pd
import numpy as np

time_index = pd.date_range('01/01/2022', periods=5, freq='M')
dataframe = pd.DataFrame(index=time_index)  # 데이터 프레임을 만들고 인덱스를 지정
print(dataframe)

dataframe['Sales'] = [1.0, 2.0, np.nan, np.nan, 5.0]  # 누락된 값이 있는 특성 생성
# data = dataframe.interpolate() # 누락된 값을 보간
data = dataframe.ffill()  # 앞쪽으로 Forward-fill
data01 = dataframe.bfill()  # 뒤쪽으로 Back-fill
data02 = dataframe.interpolate(method='quadratic')
# 보간 방향 지정
data03 = dataframe.interpolate(limit=1, limit_direction='forward')
print(data03)

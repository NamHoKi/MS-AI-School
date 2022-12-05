import numpy as np
import pandas as pd

data_strings = np.array(['12-05-2022 01:28 PM',
                        '12-06-2022 02:28 PM',
                         '12-07-2022 12:00 AM'])
# Timestamp
for data in data_strings:
    data = pd.to_datetime(data, format='%d-%m-%Y %I:%M %p')

for data in data_strings:
    data = pd.to_datetime(data, format='%d %I:%M %p', errors="ignore")
    print(data)

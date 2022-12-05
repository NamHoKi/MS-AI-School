import pandas as pd
from pytz import all_timezones
from pytz import timezone
# data = pd.Timestamp('2022-12-05 01:40:00')  # dataitem 만듬

# data_in_london = data.tz_localize(tz='Europe/London')
# print(data_in_london)

# data_in_london.tz_convert('Africa/Abidjan')
dates = pd.Series(pd.date_range('2/2/2022', periods=3, freq='M'))
# temp = dates.dt.tz_localize('Africa/Abidjan')

# print(temp)
print(all_timezones[0:2])
temp = dates.dt.tz_localize('dateutil/Aisa/Seoul')
tz = timezone('Asia/Seoul')
temp01 = dates.dt.tz_localize(tz)
print(temp01)

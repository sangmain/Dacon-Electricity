import pandas as pd
import numpy as np

test = pd.read_csv('input/test2.csv')
test['Time'] = pd.to_datetime(test['Time'])

backup = test.copy()

test = test.dropna()
# a = test.iloc[:, 0].
# print(a.get_item())
# test = test.set_index('Time')

test_dates = test['Time']
# a = test.index.values



# from sklearn.decomposition import PCA
# pca = PCA(n_components=1)
# pca.fit(test)

# x_pca = pca.transform(test)

# print("원본 형태", test.shape)
# print("축소된 형태", x_pca.shape)

# # print(x_pca)

weather = pd.read_csv('input/weather_hour.csv',encoding='euc-kr')
# print(weather)
# weather = weather.set_index('일시')
weather = weather.rename(columns={'일시' : 'Time'})
celcius = weather.iloc[:, 1]
humidity = weather.iloc[:, 4]
weather_dates = weather['Time']
# a = weather.index.values
# test_dates.get
# print(weather_dates.tolist())
# print(test_dates.tolist())
# a = list(set(test_dates).intersection(weather_dates))
# print(test_dates - weather_dates)
# print(a)
# print(weather)

print(test_dates.dtype)
print(weather_dates.dtype)

print(test_dates.astype(object))

print(weather_dates.astype(object))







# mergedStuff = pd.merge(test, weather, on=['Time'], how='inner')
# print(mergedStuff)
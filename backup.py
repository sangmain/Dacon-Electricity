import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리

weather = pd.read_csv('input/incheon_weather.csv', encoding='euc-kr')
cut_index =  weather.loc[weather['일시'] == '2017.7.1 0:00'].index
weather_cut = weather[cut_index[0]: ]
weather_cut = weather_cut.reset_index()

degree = weather.iloc[:, 2].values
humidity = weather.iloc[:, 5].values

print(degree)

x = []

for i in range(len(degree)):
    x.append([degree[i], humidity[i]])

x = np.array(x)
print(x.shape)


test = pd.read_csv('input/test2.csv')
submission = pd.read_csv("input/submission_1002.csv")

from keras.models import Sequential
from keras.layers import Dense
def build_model():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(2,)))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1))

    model.summary()
    return model

agg={}
for i in range(1, 2 ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
# for i in range(1, len(test.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
    key = test.columns[i]
    house = test.iloc[:, i]

    index = test.index()
    print(index)
    #house = house.loc[ house.isnull() == False ]
    # print(house)
    #print(house.shape)
#
    # print(start_index)
    # print(end_index)
    # diff_list = end_index - start_index -1

    # print(diff_list)
    # for ele in diff_list:
    #     if ele != 0:
    #         print(ele)

    
#     # for j in range(len(start_index)):
#     #     select_list = house[start_index[j] : end_index[j]]
#     #     print(select_list)
#     #     # house = house.drop()
#     # print(house)
    

#     print(start_index)
#     # for i in index:
#     #     null_time = i % 24 - 2
#     #     if null_time < 0:
#     #         null_time = (null_time * -1) + 20
        
#     #     house.iloc[i - null_time : (i - null_time) + 24] = None

#     # print(house)
    

        
#     # counting = house.loc[ house.iloc[:,i].isnull()==False ][ house.columns[i] ].index










# #     house = house.dropna()

# #     x_train = x[house.index]
# #     y_train = house.values
# #     y_train = y_train.reshape(y_train.shape[0], -1 )

# #     pred_data = x[-24: ]
# #     print(pred_data.shape)

# #     print(x_train.shape)
# #     print(y_train.shape)
    
# #     model = build_model()
# #     model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# #     model.fit(x_train, y_train,  batch_size= 32, epochs= 3)

# #     a = pd.DataFrame()
# #     for i in range(len(pred_data)):
# #         data = pred_data[i]
# #         data = data.reshape(-1, data.shape[0])
# #         pred = model.predict(data)

# #         print(pred, end=' ')
# #         a['X2018_7_1_'+str(i+1)+'h'] = pred[0]
# #     print('\n', key)


        
# #     for i in range(10):
# #         a['X2018_7_'+str(i+1)+'_d'] = 0. # column명을 submission 형태에 맞게 지정합니다.
        
        
# #     # 월별 예측
# #     # 일별로 예측하여 7월 ~ 11월의 일 수에 맞게 나누어 합산합니다.
# #     a['X2018_7_m'] = [0.] # 7월 
# #     a['X2018_8_m'] = [0.] # 8월
# #     a['X2018_9_m'] = [0.] # 9월
# #     a['X2018_10_m'] = [0.] # 10월
# #     a['X2018_11_m'] = [0.] # 11월
# #     a['meter_id'] = key 
# #     agg[key] = a[submission.columns.tolist()]
# #     print(key)

# # print('---- Modeling Done ----')
# # output1 = pd.concat(agg, ignore_index=False)
# # output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
# # output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
# # output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
# # output2.to_csv('hour.csv', index=False)
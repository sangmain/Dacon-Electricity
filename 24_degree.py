import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
from pandas import DataFrame #데이터 전처리 
import sys
import math
x_shape = 24 * 3
y_shape = 24 * 1
def weather_preprocess(weather):
    cut_index =  weather.loc[weather['일시'] == '2017.7.1 0:00'].index
    weather_cut = weather[cut_index[0]: ]
    weather_cut = weather_cut.reset_index()
    degree = weather.iloc[:, 2]
    humidity = weather.iloc[:, 5]

    degree = degree.fillna(method='ffill')
    humidity = humidity.fillna(method='ffill')

    # index = degree.loc[degree.isnull()].index

    # print(degree)
    return degree.values, humidity.values

x = []
# weather = pd.read_csv('input/incheon_weather.csv', encoding='euc-kr')
# degree, humidity = weather_preprocess(weather)

def nan_preprocess(test):
    test2 = test.copy()

    for k in range(1, len(test2.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
        test_median=test2.iloc[:,k].median() #값을 대체하는 과정에서 값이 변경 될 것을 대비해 해당 세대의 중앙값을 미리 계산하고 시작합니다.
        counting=test2.loc[ test2.iloc[:,k].isnull()==False ][ test2.columns[k] ].index

        df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )

        df2= df[ (df['count'] > 0) ] #결측치가 존재하는 부분만 추출
        df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

        for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수

            if test2.iloc[i,k]>=test_median: #현재 index에 존재하는 값이 해당 세대의 중앙 값 이상일때만 분산처리 실행
                test2.iloc[ i + 1 : i+j+1 , k] = test2.iloc[i,k] / (j+1) 
                #현재 index 및 결측치의 갯수 만큼 지정을 하여, 현재 index에 있는 값을 해당 갯수만큼 나누어 줍니다
            else:  
                pass #현재 index에 존재하는 값이 중앙 값 미만이면 pass를 실행
        if k%50==0: #for문 진행정도 확인용
                print(k,"번째 실행중")

    return test2

def fbfill_nan(test):
    test2 = test.copy()
    test2 = test2.fillna(method='ffill', limit= 4)
    test2 = test2.fillna(method='bfill', limit= 3)
    
    # for k in range(1, len(test2.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
    #     counting=test2.loc[ test2.iloc[:,k].isnull()==False ][ test2.columns[k] ].index

    #     df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )

    #     df2= df[ (df['count'] > 0) ] #결측치가 존재하는 부분만 추출
    #     df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

    #     for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수
    #         if j <= 6: 
    #             f_limit = b_limit = j / 2
    #             if f_limit % 1 != 0:
    #                 f_limit += 1
    #             test2 = test2.fillna(method='ffill', limit= int(f_limit))
    #             if int(b_limit) > 0:
    #                 test2 = test2.fillna(method='bfill', limit= int(b_limit))
            
                    
    return test2
def get_time(index):
    time = index % 24
    print(time)

    return time

def adjust_null(test):
    test2 = test.copy()
    for k in range(1, len(test2.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
        test_median=test2.iloc[:,k].median() #값을 대체하는 과정에서 값이 변경 될 것을 대비해 해당 세대의 중앙값을 미리 계산하고 시작합니다.
        counting=test2.loc[ test2.iloc[:,k].isnull()==False ][ test2.columns[k] ].index

        df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )

        df2= df[ (df['count'] > 0) ] #결측치가 존재하는 부분만 추출
        df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

        for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수
            start_index = i + 1
            end_index = i + j
            
            start_index = start_index - (start_index % 24)
            end_index = end_index + (24 -(end_index % 24))

            # print(start_index)
            # print(end_index)
            test2.iloc[start_index : end_index, k] = None

        counting=test2.loc[ test2.iloc[:,k].isnull()==False ][ test2.columns[k] ].index
        df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )
        df2= df[ (df['count'] > 24) ] #결측치가 존재하는 부분만 추출
        df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

        
        # for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수
        #     index_list = np.arange(i, i + j)
        #     print(index_list)
        #     test2 = test2.iloc[:, k].drop(index_list)
            





    return test2

def drop_null(data):

    ########### 첫번째 데이터까지의 nan 값들 제거
    counting= data.loc[ data.isnull()==False].index
    index_list = np.arange(0, counting[0])
    data = data.drop(index_list)

    df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )

    df2= df[ (df['count'] > 24) ] #결측치가 존재하는 부분만 추출
    df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

    for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수
        i += 1
        index_list = np.arange(i, i + j)
        data = data.drop(index_list)

    return data
            
            
def split(data, x_size, y_size):

    x = []; y = []
    for i in range(len(data) - x_size - y_size):
        
        if math.isnan(data[i]) or math.isnan(data[i + 24]) or math.isnan(data[i+ 48]):
            continue
        elif math.isnan(data[x_size + i]):
            i += 24
        x.append(data[i: x_size + i])
        y.append(data[x_size + i: x_size + y_size + i])
        

    pred_data = data[len(data) - x_size: ]
    pred_data = pred_data.reshape(-1, pred_data.shape[0])

    return np.array(x), np.array(y), pred_data

# test = pd.read_csv('input/test.csv')
# test = nan_preprocess(test)
# print(test)
# test = fbfill_nan(test)

# test = pd.read_csv('input/test4.csv')
# test = adjust_null(test)
# test.to_csv('test5.csv', index=False)

test = pd.read_csv('input/test5.csv')

for i in range(1, len(test.columns) ):
    key = test.columns[i]
    print(key)
    data = drop_null(test.iloc[:, i+1])
    data = data.values
    

    x, y, pred = split(data, x_shape, y_shape)

    print(x.shape)
    print(y.shape)
    print(pred.shape)

    from sklearn.utils import shuffle
    x, y = shuffle(x, y, random_state = 30)



    


# for i in range(len(degree)):
#     x.append([degree[i], humidity[i]])

# x = np.array(x)
# print(x.shape)


# submission = pd.read_csv("input/submission_1002.csv")

from keras.models import Sequential
from keras.layers import Dense, Conv1D
def build_model():
    model = Sequential()
    model.add(Conv1d(32, 2,)
    model.summary()
    return model

# agg={}
# for i in range(1, 2 ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
# # for i in range(1, len(test.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
#     key = test.columns[i]
#     house = test.iloc[:, i]

#     index = test.index()
#     print(index)
#     #house = house.loc[ house.isnull() == False ]
#     # print(house)
#     #print(house.shape)
# #
#     # print(start_index)
#     # print(end_index)
#     # diff_list = end_index - start_index -1

#     # print(diff_list)
#     # for ele in diff_list:
#     #     if ele != 0:
#     #         print(ele)

    
# #     # for j in range(len(start_index)):
# #     #     select_list = house[start_index[j] : end_index[j]]
# #     #     print(select_list)
# #     #     # house = house.drop()
# #     # print(house)
    

# #     print(start_index)
# #     # for i in index:
# #     #     null_time = i % 24 - 2
# #     #     if null_time < 0:
# #     #         null_time = (null_time * -1) + 20
        
# #     #     house.iloc[i - null_time : (i - null_time) + 24] = None

# #     # print(house)
    

        
# #     # counting = house.loc[ house.iloc[:,i].isnull()==False ][ house.columns[i] ].index










# # #     house = house.dropna()

# # #     x_train = x[house.index]
# # #     y_train = house.values
# # #     y_train = y_train.reshape(y_train.shape[0], -1 )

# # #     pred_data = x[-24: ]
# # #     print(pred_data.shape)

# # #     print(x_train.shape)
# # #     print(y_train.shape)
    
# # #     model = build_model()
# # #     model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# # #     model.fit(x_train, y_train,  batch_size= 32, epochs= 3)

# # #     a = pd.DataFrame()
# # #     for i in range(len(pred_data)):
# # #         data = pred_data[i]
# # #         data = data.reshape(-1, data.shape[0])
# # #         pred = model.predict(data)

# # #         print(pred, end=' ')
# # #         a['X2018_7_1_'+str(i+1)+'h'] = pred[0]
# # #     print('\n', key)


        
# # #     for i in range(10):
# # #         a['X2018_7_'+str(i+1)+'_d'] = 0. # column명을 submission 형태에 맞게 지정합니다.
        
        
# # #     # 월별 예측
# # #     # 일별로 예측하여 7월 ~ 11월의 일 수에 맞게 나누어 합산합니다.
# # #     a['X2018_7_m'] = [0.] # 7월 
# # #     a['X2018_8_m'] = [0.] # 8월
# # #     a['X2018_9_m'] = [0.] # 9월
# # #     a['X2018_10_m'] = [0.] # 10월
# # #     a['X2018_11_m'] = [0.] # 11월
# # #     a['meter_id'] = key 
# # #     agg[key] = a[submission.columns.tolist()]
# # #     print(key)

# # # print('---- Modeling Done ----')
# # # output1 = pd.concat(agg, ignore_index=False)
# # # output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
# # # output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
# # # output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
# # # output2.to_csv('hour.csv', index=False)

import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
from pandas import DataFrame #데이터 전처리 
import sys
import math


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



def adjust_null(test):
    test2 = test.copy()
    for k in range(1, len(test2.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
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

        # counting=test2.loc[ test2.iloc[:,k].isnull()==False ][ test2.columns[k] ].index
        # df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )
        # df2= df[ (df['count'] > 24) ] #결측치가 존재하는 부분만 추출
        # df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

        
        # for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수
        #     index_list = np.arange(i, i + j)
        #     print(index_list)
        #     test2 = test2.iloc[:, k].drop(index_list)
            





    return test2

def drop_null(data):

    ########### 첫번째 데이터까지의 nan 값들 제거
    counting = data.loc[ data.isnull()==False].index
    first_data_idx = counting[0]
    end_index = first_data_idx + (24 -(first_data_idx % 24))
    # data.iloc[counting[0] : end_index, k] = None

    index_list = np.arange(0, end_index)

    data = data.drop(index_list)

    df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )

    df2= df[ (df['count'] > 24) ] #결측치가 존재하는 부분만 추출
    df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

    for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수
        i += 1
        index_list = np.arange(i, i + j)
        data = data.drop(index_list)

    return data
            
def drop_dummy(test):
    test2 = test.copy()
    for k in range(1, len(test2.columns) ): 
        is_dummy = not math.isnan(test2.iloc[0, k]) 

        if is_dummy:
            test2.iloc[1394:3468, k] = None

    return test2



def split(data, x_size, y_size, gap=1, debug=False):

    x = []; y = []; batch = []

    index = int((len(data) - x_size - y_size) / gap) + 1

    for i in range(index):
        batch = data[i * gap : i * gap + x_size + y_size]
        if debug:
            print(batch, '\n')
            print(batch.shape)
        if np.isnan(batch).any():
            continue
        
        x.append(batch[0:x_size])
        y.append(batch[-y_size:])

    pred_data = data[len(data) - x_size: ]
    pred_data = pred_data.reshape(-1, pred_data.shape[0])

    return np.array(x), np.array(y), pred_data



def split2(data, data2, x_size, y_size,  gap=1, debug=False):

    x = []; y = [];
    x2 = []; y2 = [] 
    print(data.shape)
    print(data2.shape)

    index = int((len(data) - x_size - y_size) / gap) + 1
    padding = np.zeros(24)
    for i in range(index):
        # i = index - 1
        batch = data[i * gap : i * gap + x_size + y_size] 
        batch2 = data2[i * gap : i * gap + x_size + y_size]

        
        if np.isnan(batch).any():
            print('null')
            continue


        # batch = np.append(batch, padding)

        if debug:
            # print(batch, '\n')
            pass
            print(batch)
            print(batch.shape)

            print(batch2.shape)
            print(batch2)
            print(i)
            return
        seq = batch[0:x_size-y_size]
        seq =  np.append(seq, padding)

        x.append(seq)
        y.append(batch[-y_size:])

        x2.append(batch2[0:x_size])
        y2.append(batch2[-y_size:])


    x = np.array(x); x2 = np.array(2); y = np.array(y); y2 = np.array(y2);
    print(x2.shape)
    print(y2.shape)
    x = x.reshape(x.shape[0], x.shape[1], -1); x2 = x2.reshape(x.shape[0], x.shape[1], -1)
    y = y.reshape(y.shape[0], y.shape[1], -1); y2 = y2.reshape(y.shape[0], y.shape[1], -1)

    
    x = np.concatenate((x, x2), axis=1)
    y= np.concatenate((y, y2), axis=1)

    print(x.shape)
    # pred_data = data[len(data) - x_size: ]
    # pred_data = pred_data.reshape(-1, pred_data.shape[0])

    # if data2 != None:
    #     return np.array(x), np.array(y), pred_data,  np.array(x2), np.array(y2)


    # return np.array(x), np.array(y), pred_data
    


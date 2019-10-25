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
    test2 = test2.fillna(method='ffill', limit= 1)
    # test2 = test2.fillna(method='bfill', limit= 4)
    
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

def split(data, x_size, y_size):

    x = []; y = []
    cnt = 0
    index = len(data) - x_size - y_size + 1
    # print(len(data))
    index = int((len(data) - x_size - y_size) / y_size)
    for i in range(index):
        if cnt > 0:
            cnt -= 1

            # print(cnt)
            continue
            

        # if math.isnan(data[i]) or math.isnan(data[i + 24]) or math.isnan(data[i+ 48]):
        #     continue
        if math.isnan(data[x_size + i * 24]):
            cnt = x_size / 24
            continue
            # i += 1
        x.append(data[i * 24: x_size + i * 24])
        y.append(data[x_size + i * 24: x_size + y_size + i * 24])

        

    pred_data = data[len(data) - x_size: ]
    pred_data = pred_data.reshape(-1, pred_data.shape[0])

    return np.array(x), np.array(y), pred_data


from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM

def build_model():

    ##### LSTM
    # model = Sequential()
    # model.add(LSTM(50, input_shape=(x_shape, 1)))
    # model.add(Dense(50))
    # model.add(Dense(y_shape, activation='linear'))


    ##### Conv1D
    model = Sequential()
    model.add(Conv1D(128, 2, activation='relu', input_shape=(x_shape, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(y_shape, activation='linear'))

    return model



def data_summary(shape_list):

    x = np.array(shape_list)
    s = pd.Series(x)
    print(s)

    print(s.describe())

if __name__ == '__main__':

    x_shape = 24 * 7
    y_shape = 24 * 1
    shape_list = []

    test = pd.read_csv('input/test.csv')
    
    # test = pd.read_csv('input/ordered_hour.csv')
    submission = pd.read_csv("input/submission_1002.csv")

    # # churi
    # test = nan_preprocess(test)
    # test.to_csv("input/nan_pre.csv", index=False)
    # test = drop_dummy(test)
    # test.to_csv("input/drop_dummy.csv", index=False)

    test = pd.read_csv('input/drop_dummy.csv')


    test = fbfill_nan(test)
    # test.to_csv("input/nan_filled.csv", index=False)
    test = adjust_null(test)
    # test.to_csv("input/order_hour.csv", index=False)


    agg={}
    for i in range(1, len(test.columns) ):
        key = test.columns[i]
        data = drop_null(test.iloc[:, i])
        # print(data.loc[data.iloc[:] == False].index)
        data = data.values
        

        x, y, pred_data = split(data, x_shape, y_shape)
        shape_list.append(x.shape[0])

        # df = pd.DataFrame(list(zip(x, y)), columns=['X_train', 'Y_train'])
        # df.to_csv("data.csv")
        # sys.exit()

        x = x.reshape(x.shape[0], x.shape[1], -1 )
        pred_data = pred_data.reshape(pred_data.shape[0], pred_data.shape[1], -1 )

        # print(x.shape)
        # print(y.shape)
        # print(pred_data.shape)

        from sklearn.utils import shuffle
        x, y = shuffle(x, y, random_state = 30)

        test_size = 5
        x_train = x[:-test_size, :, :]
        y_train = y[:-test_size, :]

        x_test = x[-test_size : , :, :]
        y_test = y[-test_size : , :]
        
        # print(x_train.shape)
        # print(y_train.shape)

        # print(x_test.shape)
        # print(y_test.shape)
        from keras.callbacks import EarlyStopping

        model = build_model()
        model.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
        model.fit(x, y, batch_size=15, epochs=20, callbacks=[early_stopping])
        # if x.shape[0] < 100:
        #     early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
        #     model.fit(x, y, batch_size=15, epochs=20, callbacks=[early_stopping])
        # else :
        #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
        #     model.fit(x_train, y_train, batch_size=15, epochs=20, callbacks=[early_stopping], validation_data=(x_test, y_test))


        a = pd.DataFrame()
        pred = model.predict(pred_data)
        for j in range(24):
            a['X2018_7_1_'+str(j+1)+'h'] = [pred[0][j]]
            
        for j in range(10):
            a['X2018_7_'+str(j+1)+'_d'] = 0. # column명을 submission 형태에 맞게 지정합니다.
            
            
        # 월별 예측
        # 일별로 예측하여 7월 ~ 11월의 일 수에 맞게 나누어 합산합니다.
        a['X2018_7_m'] = [0.] # 7월 
        a['X2018_8_m'] = [0.] # 8월
        a['X2018_9_m'] = [0.] # 9월
        a['X2018_10_m'] = [0.] # 10월
        a['X2018_11_m'] = [0.] # 11월
        a['meter_id'] = key 

    
        agg[key] = a[submission.columns.tolist()]
        print(key)
        print("current_index: ", i)

        del model
        del data
        del x, y, x_train, y_train, x_test, y_test

    data_summary(shape_list)

    print('---- Modeling Done ----')
    output1 = pd.concat(agg, ignore_index=False)
    output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
    output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
    output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
    output2.to_csv('prediction.csv', index=False)
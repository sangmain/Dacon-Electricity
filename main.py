import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
from pandas import DataFrame #데이터 전처리 
import sys
import math
import preprocess as pp
from data_explanation import *

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout

def build_model():

    #### LSTM
    model = Sequential()
    model.add(LSTM(100, return_sequences=False, input_shape=(x_shape, 1)))
    model.add(Dropout(0.25))
    model.add(Dense(y_shape, activation='linear'))


    # ##### Conv1D
    # model = Sequential()
    # model.add(Conv1D(128, 2, activation='relu', input_shape=(x_shape, 1)))
    # model.add(MaxPooling1D(2))
    # model.add(Conv1D(64, 2, activation='relu'))
    # model.add(MaxPooling1D(2))
    # model.add(Flatten())
    # model.add(Dense(y_shape, activation='linear'))

    return model
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':

    x_shape = 24*5
    y_shape = 24 * 1
    underfit_key = []
    data_shape = []
    
    weather = pd.read_csv("input/incheon_weather.csv", encoding="euc-kr")
    
    test = pd.read_csv('input/test.csv')
    test = pp.nan_preprocess(test)
    test.to_csv("input/nan_pre.csv", index=False)


    submission = pd.read_csv("input/submission_1002.csv")

    # # 전처리
    test = pd.read_csv('input/nan_pre.csv')
    test = pp.drop_dummy(test)
    test = pp.fbfill_nan(test)
    test = pp.adjust_null(test)
    try:
        test.to_csv("input/test_result.csv", index=False)
    except:
        print("Save Failed. Permission Denied \n\n")
        pass


    agg={}

# index = weather.loc[weather.iloc[:, 1] == ].index

    weather['Time']=pd.to_datetime(weather['일시']) 
    test['Time']=pd.to_datetime(test['Time']) 
    for i in range(1, len(test.columns) ):
        key = test.columns[i]
        data = test.iloc[:, i]
        idx = data.index

        data = pp.drop_null(data)
        
        data = data.values 
        data = data.astype('float32')
        data = np.reshape(data, (-1, 1))
        scaler = MinMaxScaler(feature_range=(0, 1))
        # data = scaler.fit_transform(data)
        data = data.reshape(-1)
        print(weather.loc[weather.iloc[1, 1]])
        start_time = test.iloc[idx[0], 0]
        print(start_time)
        
        weather = weather[weather.iloc[: : , :]]
        degree, __ = weather_preprocess(weather)
        scaler_degree = MinMaxScaler()
        degree = degree.reshape(-1, 1)
        # degree = scaler_degree.fit_transform(degree)
        degree = degree.reshape(-1)
     

        split2(data, degree, x_shape, y_shape, gap=1, debug=True)

        sys.exit()

    #     data_shape.append(x.shape[0])

    #     x = x.reshape(x.shape[0], x.shape[1], -1 )
    #     pred_data = pred_data.reshape(pred_data.shape[0], pred_data.shape[1], -1 )
    #     y=y.squeeze()
    #     # print(x.shape)
    #     # print(y.shape)
    #     # print(pred_data.shape)


    #     from sklearn.utils import shuffle
    #     x, y = shuffle(x, y, random_state = 30)
    #     from sklearn.model_selection import KFold
    #     kf = KFold(n_splits=15)
    #     for train_index, test_index in kf.split(x):
    #         x_train, x_test = x[train_index], x[test_index] 
    #         y_train, y_test = y[train_index], y[test_index]


    #     # test_size = 50
    #     # x_train = x[:-test_size, :, :]
    #     # y_train = y[:-test_size, :]

    #     # x_test = x[-test_size : , :, :]
    #     # y_test = y[-test_size : , :]
    #     print(x.shape)
    #     print(x_train.shape)
    #     print(x_test.shape)

    #     from keras.callbacks import EarlyStopping

    #     model = build_model()
    #     model.compile(optimizer='adam', loss='mse')

    #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    #     model.fit(x_train, y_train, batch_size=512, epochs=2, callbacks=[early_stopping], validation_data=(x_test, y_test), verbose= 0)
    #     loss = model.evaluate(x_test, y_test, verbose= 0)
    #     print(loss)
    #     if loss > 1.0:
    #         print(loss)
    #         underfit_key.append(key)

    #     a = pd.DataFrame()
    #     pred = model.predict(pred_data)
    #     pred = scaler.inverse_transform(pred)

    #     for j in range(24):
    #         if  pred[0][j] < 0:
    #             print("found value less than 0")
    #             sys.exit()
    #         a['X2018_7_1_'+str(j+1)+'h'] = [pred[0][j]]
            
    #     for j in range(10):
    #         a['X2018_7_'+str(j+1)+'_d'] = 0. # column명을 submission 형태에 맞게 지정합니다.
            
            
    #     # 월별 예측
    #     # 일별로 예측하여 7월 ~ 11월의 일 수에 맞게 나누어 합산합니다.
    #     a['X2018_7_m'] = [0.] # 7월 
    #     a['X2018_8_m'] = [0.] # 8월
    #     a['X2018_9_m'] = [0.] # 9월
    #     a['X2018_10_m'] = [0.] # 10월
    #     a['X2018_11_m'] = [0.] # 11월
    #     a['meter_id'] = key 

    
    #     agg[key] = a[submission.columns.tolist()]
    #     print(key)
    #     print("current_index: ", i)

    #     del model
    #     del data
    #     del x, y
    #     print('\n\n\n')

    # data_summary(data_shape)

    # print('---- Modeling Done ----')
    # print("Model with loss higher than 1.0\n", underfit_key)
    # print("count", len(underfit_key))
    # output1 = pd.concat(agg, ignore_index=False)
    # output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
    # output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
    # output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
    # output2.to_csv('prediction.csv', index=False)






import matplotlib.pyplot as plt # 데이터 시각화
import itertools
from datetime import datetime, timedelta # 시간 데이터 처리
from statsmodels.tsa.arima_model import ARIMA # ARIMA 모델
# %matplotlib inline
x_shape = 7 * 24
y_shape = 1 * 24
test = pd.read_csv("input/test2.csv")
submission = pd.read_csv("input/submission_1002.csv")

print(test.shape)
cut_index = test.loc[test['Time'] == '2018.4.19 0:00'].index
test_cut = test[cut_index[0]: ]

def split(data, x_size, y_size):

    x = []; y = []
    for i in range(len(data) - x_size - y_size):
        
        x.append(data[i: x_size + i])
        y.append(data[x_size + i: x_size + y_size + i])
        
    pred_data = data[len(data) - x_size: ]
    pred_data = pred_data.reshape(-1, pred_data.shape[0])



    return np.array(x), np.array(y), pred_data

    

from keras.models import Sequential
from keras.layers import Dense, LSTM

agg={}
for i in range(1, len(test_cut.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
    key = test_cut.columns[i]
    # test_median=test_cut.iloc[:,i].mean()
    # test_cut.iloc[:, i] = test_cut.iloc[:,i].fillna(test_median)
    test_cut.iloc[:, i] = test_cut.iloc[:,i].fillna(method='ffill')

    row = test_cut.iloc[:, i].values
    # print(row)
    x, y, pred_data = split(row, x_shape, y_shape)
    # print(x.shape)
    # print(y.shape)
    # print(pred_data.shape)

    x = x.reshape(x.shape[0], x.shape[1], -1)
    y = y.reshape(y.shape[0], y.shape[1])
    pred_data = pred_data.reshape(pred_data.shape[0], pred_data.shape[1], -1)

    model = Sequential()

    model.add(LSTM(50, return_sequences=False, input_shape=(x_shape, 1)))

    model.add(Dense(50))
    model.add(Dense(y_shape, activation='linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    model.summary()
    
    model.fit(x, y,  batch_size= 32 , epochs=2)

    a = pd.DataFrame()
    fcst = model.predict(pred_data)
    for i in range(24):
        a['X2018_7_1_'+str(i+1)+'h']=[fcst[0][i]]

    for i in range(10):
        a['X2018_7_'+str(i+1)+'_d'] = None # column명을 submission 형태에 맞게 지정합니다.
    
    
    # 월별 예측
    # 일별로 예측하여 7월 ~ 11월의 일 수에 맞게 나누어 합산합니다.
    a['X2018_7_m'] = [None] # 7월 
    a['X2018_8_m'] = [None] # 8월
    a['X2018_9_m'] = [None] # 9월
    a['X2018_10_m'] = [None] # 10월
    a['X2018_11_m'] = [None] # 11월
    a['meter_id'] = key 
    agg[key] = a[submission.columns.tolist()]
    print(key)
print('---- Modeling Done ----')


output1 = pd.concat(agg, ignore_index=False)
output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
output2.to_csv('hour.csv', index=False)
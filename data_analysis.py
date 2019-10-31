import numpy as np


import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf

# 데이터를 읽어들인다
df_ori = pd.read_csv("input/test.csv")
print(df_ori)

for k in range(1, len(df_ori.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킨다.
    key = df_ori.columns[k]

    df=df_ori.loc[:,['Time',key]] # 한 가구만 가져온다
    print(df)
    df = df.dropna()
    #시간을 연도, 4분기, 달, 일로 보여준다
    df['Time']=pd.to_datetime(df['Time']) 
    df['year'] = df['Time'].apply(lambda x: x.year)
    df['quarter'] = df['Time'].apply(lambda x: x.quarter)
    df['month'] = df['Time'].apply(lambda x: x.month)
    df['day'] = df['Time'].apply(lambda x: x.day)
    df=df.loc[:,['Time',key, 'year','quarter','month','day']]
    df.sort_values('Time', inplace=True, ascending=True) # 시간 순으로 재배열
    df = df.reset_index(drop=True) # 인덱스 초기화

    #주일과 주말까지 추가한다
    df["weekday"]=df.apply(lambda row: row["Time"].weekday(),axis=1)
    df["weekday"] = (df["weekday"] < 5).astype(int)

    print('Number of rows and columns after removing missing values:', df.shape)
    print('The time series starts from: ', df.Time.min())
    print('The time series ends on: ', df.Time.max())
    print()

    stat, p = stats.normaltest(df[key])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Data looks Gaussian (fail to reject H0)')
    else:
        print('Data does not look Gaussian (reject H0)')


    sns.distplot(df[key]);
    print( 'Kurtosis of normal distribution: {}'.format(stats.kurtosis(df[key])))
    print( 'Skewness of normal distribution: {}'.format(stats.skew(df[key])))

    plt.show()


    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.subplots_adjust(wspace=0.2)
    sns.boxplot(x="year", y=key, data=df)
    plt.xlabel('year')
    plt.title('Box plot of Yearly Energy Consumed')
    sns.despine(left=True)
    plt.tight_layout()
    plt.subplot(1,2,2)
    sns.boxplot(x="quarter", y=key, data=df)
    plt.xlabel('quarter')
    plt.title('Box plot of Quarterly Energy Consumed')
    sns.despine(left=True)
    plt.tight_layout();


    plt.show()

    dic={0:'Weekend',1:'Weekday'}
    df['Day'] = df.weekday.map(dic)
        
    plt1=sns.factorplot('year', key ,hue='Day',
                        data=df, size=4, aspect=1.5, legend=False)                                                                                                                                                                                                                                                                                                                                             
    plt.title('Factor Plot of Energy Consumation by Weekend/Weekday')                                                             
    plt.tight_layout()                                                                                                                  
    sns.despine(left=True, bottom=True) 
    plt.legend(loc='upper right');

    plt.show()


    # df2=df1.resample('D', how=np.mean)

    # def test_stationarity(timeseries):
    #     rolmean = timeseries.rolling(window=30).mean()
    #     rolstd = timeseries.rolling(window=30).std()
        
    #     plt.figure(figsize=(14,5))
    #     sns.despine(left=True)
    #     orig = plt.plot(timeseries, color='blue',label='Original')
    #     mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    #     std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    #     plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    #     plt.show()
        
    #     print ('<Results of Dickey-Fuller Test>')
    #     dftest = adfuller(timeseries, autolag='AIC')
    #     dfoutput = pd.Series(dftest[0:4],
    #                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    #     for key,value in dftest[4].items():
    #         dfoutput['Critical Value (%s)'%key] = value
    #     print(dfoutput)
    # test_stationarity(df2[key].dropna())

    input()

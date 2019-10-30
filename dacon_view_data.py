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

df5 = pd.read_csv("input/test_result.csv")
for k in range(1, len(df5.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
    key = df5.columns[k]

    df=df5.loc[:,['Time',key]]
    df = df.dropna()
    df['Time']=pd.to_datetime(df['Time']) 
    df['year'] = df['Time'].apply(lambda x: x.year)
    df['quarter'] = df['Time'].apply(lambda x: x.quarter)
    df['month'] = df['Time'].apply(lambda x: x.month)
    df['day'] = df['Time'].apply(lambda x: x.day)
    df=df.loc[:,['Time',key, 'year','quarter','month','day']]
    df.sort_values('Time', inplace=True, ascending=True)
    df = df.reset_index(drop=True)
    df["weekday"]=df.apply(lambda row: row["Time"].weekday(),axis=1)
    df["weekday"] = (df["weekday"] < 5).astype(int)
    print('Number of rows and columns after removing missing values:', df.shape)
    print('The time series starts from: ', df.Time.min())
    print('The time series ends on: ', df.Time.max())


    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.subplots_adjust(wspace=0.2)
    sns.boxplot(x="year", y=key, data=df)
    plt.xlabel('year')
    plt.title('Box plot of Yearly Global Active Power')
    sns.despine(left=True)
    plt.tight_layout()
    plt.subplot(1,2,2)
    sns.boxplot(x="quarter", y=key, data=df)
    plt.xlabel('quarter')
    plt.title('Box plot of Quarterly Global Active Power')
    sns.despine(left=True)
    plt.tight_layout();


    plt.show()

    df1=df.loc[:,['Time',key]]
    df1.set_index('Time',inplace=True)

    fig = plt.figure(figsize=(18,16))
    fig.subplots_adjust(hspace=.4)
    ax1 = fig.add_subplot(5,1,1)
    ax1.plot(df1[key].resample('D').mean(),linewidth=1)
    ax1.set_title('Mean Global active power resampled over day')
    ax1.tick_params(axis='both', which='major')

    ax2 = fig.add_subplot(5,1,2, sharex=ax1)
    ax2.plot(df1[key].resample('W').mean(),linewidth=1)
    ax2.set_title('Mean Global active power resampled over week')
    ax2.tick_params(axis='both', which='major')

    ax3 = fig.add_subplot(5,1,3, sharex=ax1)
    ax3.plot(df1[key].resample('M').mean(),linewidth=1)
    ax3.set_title('Mean Global active power resampled over month')
    ax3.tick_params(axis='both', which='major')

    ax4  = fig.add_subplot(5,1,4, sharex=ax1)
    ax4.plot(df1[key].resample('Q').mean(),linewidth=1)
    ax4.set_title('Mean Global active power resampled over quarter')
    ax4.tick_params(axis='both', which='major')

    ax5  = fig.add_subplot(5,1,5, sharex=ax1)
    ax5.plot(df1[key].resample('A').mean(),linewidth=1)
    ax5.set_title('Mean Global active power resampled over year')
    ax5.tick_params(axis='both', which='major');

    plt.show()

    dic={0:'Weekend',1:'Weekday'}
    df['Day'] = df.weekday.map(dic)
    a=plt.figure(figsize=(9,4)) 
    plt1=sns.boxplot('year',key,hue='Day',width=0.6,fliersize=3,
                        data=df)                                                                                                                                                                                                                                                                                                                                                 
    a.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
    sns.despine(left=True, bottom=True) 
    plt.xlabel('')
    plt.tight_layout()                                                                                                                  
    plt.legend().set_visible(False);
    plt1=sns.factorplot('year', key,hue='Day',
                    data=df, size=4, aspect=1.5, legend=False)                                                                                                                                                                                                                                                                                                                                             
    plt.title('Factor Plot of Global active power by Weekend/Weekday')                                                             
    plt.tight_layout()                                                                                                                  
    sns.despine(left=True, bottom=True) 
    plt.legend(loc='upper right');

    plt.show()


    df2=df1.resample('D', how=np.mean)

    def test_stationarity(timeseries):
        rolmean = timeseries.rolling(window=30).mean()
        rolstd = timeseries.rolling(window=30).std()
        
        plt.figure(figsize=(14,5))
        sns.despine(left=True)
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')

        plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
        plt.show()
        
        print ('<Results of Dickey-Fuller Test>')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                            index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    test_stationarity(df2[key].dropna())
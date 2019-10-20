import numpy as np #데이터 전처리
import pandas as pd #데이터 전처리
from pandas import DataFrame #데이터 전처리 

import matplotlib.pyplot as plt #데이터 시각화
import seaborn as sns #데이터 시각화

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

print(train.shape)
print(test.shape)

test2 = test.copy()  #원본 데이터 보존을 위한 데이터 복사

X303 = test2.loc[ test2.iloc[:,2].isnull() == False ]['X303'].index # X303세대에서 NaN이 아닌 index만을 추출

df = DataFrame( list( zip( X303[:-1], X303[1:]-X303[:-1] -1 ) ) , columns=[ 'index', 'count' ]) # index 값과 NaN 갯수들을 DataFrame 형태로 생성함

df.loc[df.shape[0]+1] = (X303[-1] , 8759 - X303[-1]) #마지막 값들이 결측일 경우를 추가하기 위한 코드
# 마지막 값이 결측이 아니면 값 0입니다. #마지막 값이 결측이라면 해당 갯수만큼 반영이 됩니다.

df = df.sort_values( ["count"], ascending=[False]) #결측치 갯수를 기준으로 내림차순으로 DataFrame을 정렬함


for k in range(1, len(test2.columns) ): #시간을 제외한 1열부터 마지막 열까지를 for문으로 작동시킵니다.
    test_median=test2.iloc[:,k].median() #값을 대체하는 과정에서 값이 변경 될 것을 대비해 해당 세대의 중앙값을 미리 계산하고 시작합니다.
    counting=test2.loc[ test2.iloc[:,k].isnull()==False ][ test2.columns[k] ].index

    df=DataFrame( list( zip( counting[:-1], counting[1:] - counting[:-1] -1  ) ), columns=['index','count'] )

    df2= df[ (df['count'] > 0) ] #결측치가 존재하는 부분만 추출
    df2=df2.reset_index(drop=True) #기존에 존재하는 index를 초기화 하여 이후 for문에 사용함

    for i,j in zip( df2['index'], df2['count'] ) : # i = 해당 세대에서 값이 존재하는 index, j = 현재 index 밑의 결측치 갯수
        if test2.iloc[i,k]>=test_median: #현재 index에 존재하는 값이 해당 세대의 중앙 값 이상일때만 분산처리 실행
            test2.iloc[ i : i+j+1 , k] = test2.iloc[i,k] / (j+1) 
            #현재 index 및 결측치의 갯수 만큼 지정을 하여, 현재 index에 있는 값을 해당 갯수만큼 나누어 줍니다
        else:  
            pass #현재 index에 존재하는 값이 중앙 값 미만이면 pass를 실행
    if k%50==0: #for문 진행정도 확인용
            print(k,"번째 실행중")

test2.to_csv('test2.csv',index=False) #결측치 대체 작업 이후 csv 파일 내보내기
import seaborn as sns
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

test = pd.read_csv("input/order_hour.csv")
for i in range(1, len(test.columns)):
    key = test.columns[i]

    data = test.iloc[:, i]
    data = data.dropna()
    data = data.values
    sns.kdeplot(data)
    plt.title("data distribution")
    plt.show()

    data = data.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    data = data.reshape(-1)
    scaled_data = scaled_data.reshape(-1)

    print(data)
    print(scaled_data)
    sns.kdeplot(scaled_data)
    plt.title("scaled_data distribution")
    plt.show()


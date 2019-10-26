import numpy as np  
import pandas as pd

def find_data(data, cmp_data):
    # for i in range(data.shape[0]):
    merged_data = np.intersect1d(data, cmp_data)
    print(len(merged_data))
    if len(merged_data) >= len(cmp_data) / 2:
        return True

        
    return False
        

def view_data(x, y):
    print("*********** DATA STATISTICS ***********")
    print('x shape: ', x.shape)
    print('y shape: ', y.shape)

    print('Does X contain NAN: ', np.isnan(x).any())
    print('Does Y contain NAN: ', np.isnan(y).any())
    
    index = 0
    # print('X[%d]\n' % index, x[index])
    # print('Y[%d]\n' % index, y[index])

    index = 1
    # print('X[%d]\n' % index, x[index])
    # print('Y[%d]\n' % index, y[index])



def is_bad_data(data, gap=1):
    is_error = data[0, 1] != data[0+gap, 0]
    return is_error        
        
        # print(data[i])
    

def data_summary(shape_list):

    x = np.array(shape_list)
    s = pd.Series(x)
    print(s)

    print(s.describe())


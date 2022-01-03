import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_kddcup99
from miss_generator import generate_missing_value_mask

def load_iris_miss(miss_rate, complete_ratio): 
    '''
    Desciption: 
    Use all the data includes label as imputation target, thus there will be 5 attributes
    150 tuples/records/rows, 5 attributes/features/columns(includes label)
    Output: 
    Original X and Y in X_Y, also X_Y_miss imcludes missing values. 
    Ref: 
    numpy array add column: https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
    '''
    X,Y = load_iris(return_X_y=True)
    X_Y = np.c_[X,Y]
    
    miss_mask = generate_missing_value_mask(
        X_Y.shape[0], X_Y.shape[1], miss_rate, complete_ratio
    )
    X_Y_miss = X_Y.copy()
    X_Y_miss[miss_mask] = np.NaN

    
    # print(X.shape)
    # print(Y.shape)
    # print(X_Y.shape)
    # print(miss_mask.shape)
    # print(X_Y_miss.shape)
    # print("Number of miss items: ", np.count_nonzero(np.isnan(X_Y_miss)))
    return X_Y, X_Y_miss

if __name__=="__main__": 
    # 75 missing items, 75 incomplete tuples
    # load_iris_miss(0.1, 0.5)
    # 112 missing items, 100 incomplete tuples
    load_iris_miss(0.15, 1/3)

    # TODO: Check if the below function calls works coorectly/fine
    # 75 missing items, 75 incomplete tuples
    # load_iris_miss(0.2, 0.5)
    # 75 missing items, 75 incomplete tuples
    # load_iris_miss(0.2, 1/3)
    # 75 missing items, 75 incomplete tuples
    # load_iris_miss(0.2, 1/6)
    # 75 missing items, 75 incomplete tuples
    # load_iris_miss(0.3, 0.5)
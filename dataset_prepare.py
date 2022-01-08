import numpy as np
import pandas as pd
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
    The missing value mask will output. 
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
    return X_Y, X_Y_miss, miss_mask

def load_KDDCUP99_miss(miss_rate, complete_ratio): 
    '''
    Only choose 10% of the dataset, and not including the labels
    Ref: convert categorical columns to integer codes: https://stackoverflow.com/questions/32011359/convert-categorical-data-in-pandas-dataframe
    '''
    X,Y = fetch_kddcup99(return_X_y=True, percent10=True)
    cat_idx = [1, 2, 3, 6, 11, 20, 21]
    n_cat_idx = np.arange(X.shape[1])
    n_cat_idx = n_cat_idx[~np.isin(n_cat_idx, cat_idx)]
    df = pd.DataFrame(X)
    for i in cat_idx: 
        df[i] = df[i].astype('category')
    for i in n_cat_idx: 
        df[i] = df[i].astype('float')
    # Convert categorical attributes to integer codes
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    X = df.to_numpy()

    miss_mask = generate_missing_value_mask(
        X.shape[0], X.shape[1], miss_rate, complete_ratio
    )
    X_miss = X.copy()
    X_miss[miss_mask] = np.NaN

    cat_mask = np.full(X.shape[1], False)
    cat_mask[cat_idx] = True
    print(cat_mask)
    print('KDDCUP99 prepare complete')
    return X, X_miss, miss_mask, cat_mask
    

if __name__=="__main__": 
    # Check if the below function calls works coorectly/fine

    # 75 missing items, 75 incomplete tuples
    # load_iris_miss(0.1, 0.5)
    # 112 missing items, 100 incomplete tuples
    # load_iris_miss(0.15, 1/3)
    # 150 missing items, 75 incomplete tuples
    # load_iris_miss(0.2, 0.5)
    # 150 missing items, 100 incomplete tuples
    # load_iris_miss(0.2, 1/3)
    # 150 missing items, 125 incomplete tuples
    # load_iris_miss(0.2, 1/6)
    # 225 missing items, 75 incomplete tuples
    # load_iris_miss(0.3, 0.5)

    # KDDCUP99
    load_KDDCUP99_miss(0.1, 0.5)
    load_KDDCUP99_miss(0.15, 1/3)
    load_KDDCUP99_miss(0.2, 0.5)
    load_KDDCUP99_miss(0.2, 1/3)
    load_KDDCUP99_miss(0.2, 1/6)
    load_KDDCUP99_miss(0.3, 0.5)

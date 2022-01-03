import numpy as np
import itertools

def nan_euclidean_with_categorical(X: np.array, Y: np.array, missing_values, categorical_mask): 
    '''
    Description: 
    calculate the distance between each row of X and Y
    And ignore the missing value
    Distance of categorical data is 0 if they are the same, 1 if different
    Output: 
    :Dis: same length as X and Y
    Ref: format of _pairwise_callable in sklearn: https://github.com/scikit-learn/scikit-learn/blob/ed865d7a3363a92846d7955a9bdedae2ad29542e/sklearn/metrics/pairwise.py#L1448
    Ref: Euclidean distance implementatio: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    '''
    out = np.empty((X.shape[0], Y.shape[0]), dtype="float")
    iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
    for i, j in iterator: 
        nan_mask = np.isnan(X[i]) | np.isnan(Y[j])
        print(X[i])
        print(X)
        print(X.shape)
        print(nan_mask)
        # No nan and categorical or numerical
        X_nnan_cat = X[i, ~nan_mask & categorical_mask]
        Y_nnan_cat = Y[j, ~nan_mask & categorical_mask]
        X_nnan_num = X[i, ~nan_mask & ~categorical_mask]
        Y_nnan_num = Y[j, ~nan_mask & ~categorical_mask]
        _distance = 0
        # print(X_nnan_cat)
        # print(Y_nnan_cat)
        # print(X_nnan_num)
        # print(Y_nnan_num)
        if X_nnan_cat.size is not 0: 
            _distance += np.sum(~(X_nnan_cat==Y_nnan_cat))
        if X_nnan_num.size is not 0: 
            _distance += np.sqrt(
                np.dot(X_nnan_num, X_nnan_num)-2*np.dot(X_nnan_num, Y_nnan_num)+np.dot(Y_nnan_num, Y_nnan_num)
            )
        # print(_distance)
        # print('-------')
        out[i, j]=_distance
    return out

def paired_nan_euclidean_with_categorical(X: np.array, Y: np.array, missing_values, categorical_mask): 
    nan_mask = np.isnan(X) | np.isnan(Y)
    # print(X)
    # print(X.shape)
    # print(nan_mask)
    # No nan and categorical or numerical
    X_nnan_cat = X[~nan_mask & categorical_mask]
    Y_nnan_cat = Y[~nan_mask & categorical_mask]
    X_nnan_num = X[~nan_mask & ~categorical_mask]
    Y_nnan_num = Y[~nan_mask & ~categorical_mask]
    _distance = 0
    # print(X_nnan_cat)
    # print(Y_nnan_cat)
    # print(X_nnan_num)
    # print(Y_nnan_num)
    if X_nnan_cat.size is not 0: 
        _distance += np.sum(~(X_nnan_cat==Y_nnan_cat))
    if X_nnan_num.size is not 0: 
        _distance += np.sqrt(
            np.dot(X_nnan_num, X_nnan_num)-2*np.dot(X_nnan_num, Y_nnan_num)+np.dot(Y_nnan_num, Y_nnan_num)
        )
    # print(_distance)
    # print('-------')
    return _distance

if __name__=="__main__": 
    X = np.array([[0, 1, 2, np.nan, 5], [1, np.nan, 2, 2, 2]])
    dis=nan_euclidean_with_categorical(X, X, missing_values=np.nan, categorical_mask=np.array([True, False, False, False, False]))
    dis=paired_nan_euclidean_with_categorical(
        X[0], X[1], missing_values=np.nan, categorical_mask=np.array([True, False, False, False, False])
    )
    print(dis)
import numpy as np
import scipy.stats as st
from distance_measurement import nan_euclidean_with_categorical
from dataset_prepare import load_iris_miss

def KNN_impute_voting(k, dataset, pairwise_dis_func): 
    '''
    Use KNN to impute missing values by voting. 
    Specifically for categorical attributes. 
    '''
    imputed_dataset=dataset.copy()
    dis = pairwise_dis_func(dataset, dataset)
    na_mask = np.isnan(dataset)
    # Not nan tuples in each attribute
    not_na_tuple_idx = [np.where(~na_mask[:,i])[0] for i in range(dataset.shape[1])]
    have_na_tuple_idx=np.argwhere(np.sum(na_mask, axis=1)>0).T[0]
    print(dis.shape)
    print(have_na_tuple_idx)
    for _idx in have_na_tuple_idx: 
        _dis = dis[_idx]
        # print(_dis)
        # print(dataset[_idx])
        sorted_similar_idx = np.argsort(_dis)
        sorted_similar_idx = sorted_similar_idx[sorted_similar_idx!=_idx]
        # print(sorted_similar_idx)
        na_attr_idx = np.where(np.isnan(dataset[_idx]))[0]
        # print(na_attr_idx)
        for j in na_attr_idx: 
            _similar_not_na_idx = sorted_similar_idx[np.isin(sorted_similar_idx, not_na_tuple_idx[j])]
            if _similar_not_na_idx.shape[0] > k: 
                _similar_not_na_idx = _similar_not_na_idx[:k]
            # print(_similar_not_na_idx)
            # Vote!!
            # print(dataset[_similar_not_na_idx, j])
            # print(st.mode(dataset[_similar_not_na_idx, j])[0][0])
            imputed_dataset[_idx,j] = st.mode(dataset[_similar_not_na_idx, j])[0]
    return imputed_dataset

if __name__=="__main__": 
    # Get dataset
    full_X, miss_X, miss_mask = load_iris_miss(0.2, 0.5)
    # Generate categorical mask
    # (In iris dataset the last one is categorical)
    cat_mask = np.full(full_X.shape[1], False)
    cat_mask[-1] = True

    partial_miss_X = miss_X[::10, :]
    partial_miss_mask = miss_mask[::10, :]
    # print(partial_miss_X)
    # print(partial_miss_mask)
    # print(np.sum(partial_miss_mask, axis=1))
    # print(np.sum(partial_miss_mask, axis=1)>0)
    # _idx=np.argwhere(np.sum(partial_miss_mask, axis=1)>0).T[0]
    # print(_idx)
    # print(partial_miss_X[_idx[0]])
    # print(partial_miss_mask[_idx[0]])

    KNN_impute_voting(
        5, 
        miss_X, 
        lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, cat_mask)
        )
    pass
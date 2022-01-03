import numpy as np
from distance_measurement import nan_euclidean_with_categorical
from dataset_prepare import load_iris_miss

def KNN_impute_voting(dataset, pairwise_dis_func): 
    '''
    Use KNN to impute missing values by voting. 
    Specifically for categorical attributes. 
    '''
    dis = pairwise_dis_func(dataset, dataset)
    na_mask = np.isnan(dataset)
    have_na_tuple_idx=np.argwhere(np.sum(na_mask, axis=1)>0).T[0]
    print(dis.shape)
    print(have_na_tuple_idx)
    for _idx in have_na_tuple_idx: 
        _dis = dis[_idx]
        print(_dis)
        print(dataset[_idx])
        # TODO: 
        break
        pass


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
        miss_X, 
        lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, cat_mask)
        )
    pass
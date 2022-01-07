import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from dataset_prepare import load_iris_miss
from Improved_kmeans import improved_kmeans_impute
from KNN_imputation import KNN_impute

'''
try each imputation method and plot RMSE and f1 result
'''
# draw evaluation plot
def draw_impute_eval(x, y_dict, name_title='no title', name_x='no name', final_show=True): 
    plt.figure()
    plt.title(name_title)
    plt.ylabel(name_x)
    plt.xlabel('missing and incomplete rate')
    for k in y_dict: 
        plt.plot(x, y_dict[k], label=k)
    plt.legend()
    if final_show: 
        plt.show()

# KNN imputer

# improved k-means imputer
if __name__=="__main__": 
    incomplete_miss_rate = [
        [1/2, 0.1], [1/3, 0.15], [1/2, 0.2], 
        [1/3, 0.2], [1/6, 0.2], [1/2, 0.3], 
        [1/3, 0.3], [1/6, 0.3], [1/6, 0.4]
    ]
    incomplete_miss_rate_str = [
        '1/2_0.1', '1/3_0.15', '1/2_0.2', 
        '1/3_0.2','1/6_0.2', '1/2_0.3', 
        '1/3_0.3', '1/6_0.3', '1/6_0.4'
    ]
    impute_methods = {
        'improved k-means euclidean': lambda cat_mask, miss_X: improved_kmeans_impute(
            3, 
            5, 
            cat_mask, 
            miss_X, 
            if_euclidean=True
        ), 
        'improved k-means hahalanobis': lambda cat_mask, miss_X: improved_kmeans_impute(
            3, 
            5, 
            cat_mask, 
            miss_X, 
            if_euclidean=False
        ), 
        'KNN euclidean': lambda cat_mask, miss_X: KNN_impute(
            k=5, 
            miss_data=miss_X, 
            cat_mask=cat_mask
        )
    }
    rmse_incomplete_miss = {}
    f1_incomplete_miss = {}
    for i in impute_methods: 
        rmse_incomplete_miss[i] = []
        f1_incomplete_miss[i] = []
    for _incomplete_miss_rate in incomplete_miss_rate: 
        # get data
        ## iris
        full_X, miss_X, miss_mask = load_iris_miss(_incomplete_miss_rate[1], _incomplete_miss_rate[0])
        cat_mask_iris = np.full(full_X.shape[1], False)
        cat_mask_iris[-1] = True
        ## KDDCUP 99

        # improved k-means (Euclidean and mahalanobis)
        imputed_results = {
            _impute_method_nm: \
                impute_methods[_impute_method_nm](cat_mask_iris, miss_X) for _impute_method_nm in impute_methods
        }
        # imputed_results = {
        #     'improved k-means euclidean': improved_kmeans_impute(
        #         3, 
        #         5, 
        #         cat_mask_iris, 
        #         miss_X, 
        #         if_euclidean=True
        #     ), 
        #     'improved k-means hahalanobis': improved_kmeans_impute(
        #         3, 
        #         5, 
        #         cat_mask_iris, 
        #         miss_X, 
        #         if_euclidean=False
        #     )
        # }
        # Evaluation
        for _imputed_method in imputed_results: 
            cat_mask_expand = np.repeat(cat_mask_iris[np.newaxis, :], full_X.shape[0], axis=0)
            ## RMSE
            rmse = mean_squared_error(full_X[miss_mask&~cat_mask_expand], imputed_results[_imputed_method][miss_mask&~cat_mask_expand])
            rmse = rmse**(1/2)
            ## f1
            f1 = f1_score(full_X[miss_mask&cat_mask_expand], imputed_results[_imputed_method][miss_mask&cat_mask_expand], average='micro')
            # TODO: 改成使用dictionary儲存eval result
            rmse_incomplete_miss[_imputed_method].append(rmse)
            f1_incomplete_miss[_imputed_method].append(f1)
    print(rmse_incomplete_miss)
    print(f1_incomplete_miss)
    draw_impute_eval(incomplete_miss_rate_str, rmse_incomplete_miss
        , name_title='iris', name_x='rmse', final_show=False
    )
    draw_impute_eval(incomplete_miss_rate_str, f1_incomplete_miss
        , name_title='iris', name_x='f1'
    )
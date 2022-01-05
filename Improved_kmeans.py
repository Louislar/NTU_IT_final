import numpy as np
from scipy.sparse import data
import scipy.stats as st
from scipy.spatial import distance
from sklearn.utils.extmath import weighted_mode
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from distance_measurement import nan_euclidean_with_categorical
from dataset_prepare import load_iris_miss

'''
Ref: https://github.com/bamtak/machine-learning-implemetation-python/blob/master/KMeans.ipynb
'''
np.random.seed(3)

class KMeans:
    
    def __init__(self, n_clusters=4, max_iter=1e6):
        self.K = n_clusters
        self.max_iter = max_iter
        print("max iteration", max_iter)
        
    def fit(self, X):
        self.inv_conv = np.linalg.pinv(np.cov(X.T))
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        counter=0
        while (not np.all(self.labels == self.prev_label)) and counter<self.max_iter:
            self.prev_label = self.labels
            self.labels = self.predict(X)
            self.update_centroid(X)
            counter+=1
            if counter%100==0: 
                print(counter)
        return self
        
    def predict(self, X):
        # print(np.apply_along_axis(self.compute_label_mahalanobis, 1, X))
        return np.apply_along_axis(self.compute_label, 1, X)
        return np.apply_along_axis(self.compute_label_mahalanobis, 1, X)

    def compute_label(self, x):
        return np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1)))
    
    def compute_label_mahalanobis(self, x): 
        # Min-max scale the centroid and the data point, 
        # since too small covariance will cause mahala-distance failuer
        # scalar = MinMaxScaler(feature_range=(1, 10))
        # rescale_data = scalar.fit_transform(np.vstack((x, self.centroids)))
        # x = rescale_data[0]
        # _centroids = rescale_data[1:]

        _centroids = self.centroids
        # covs = np.apply_along_axis(lambda input: np.cov(np.stack((input,x), axis=0).T), 1, _centroids)
        # inv_covs = np.linalg.pinv(covs) # inverse may not exist, so pseudo inverse is a choice
        mahala_dis = np.empty(self.K)
        # print(x)
        # print(_centroids)
        
        for i in range(len(mahala_dis)): 
            # diff_centroid = x-_centroids[i]
            # print(diff_centroid)
            # norm_diff_centroid = diff_centroid / np.linalg.norm(diff_centroid)
            # print(np.dot(np.dot(norm_diff_centroid, inv_covs[i]), norm_diff_centroid))
            # mahala_dis[i] = np.dot(np.dot(diff_centroid, self.inv_conv), diff_centroid)
            mahala_dis[i] = \
                distance.mahalanobis(x,_centroids[i],self.inv_conv)
        # print(mahala_dis)
        # exit()
        return np.argmin(mahala_dis)

    def update_centroid(self, X):
        self.centroids = np.array([np.mean(X[self.labels == k], axis=0)  for k in range(self.K)])

def imporved_kmeans_impute(k, t, cat_mask, dataset, pairwise_dis_func): 
    '''
    Input: 
    :k: k in k-means clustering
    :t: t most similar tuples/samples will used to compute the EWM's weight
    :cat_mask: categorical mask, true if the attribute is categorical
    '''
    # 1. prefill with mean and mode 
    imputed_dataset=dataset.copy()
    dataset = dataset.copy()
    na_mask = np.isnan(dataset)
    mean_n_mode = np.empty(dataset.shape[1])
    mean_n_mode[~cat_mask] = np.nanmean(dataset[:, ~cat_mask], axis=0)
    mean_n_mode[cat_mask] = st.mode(dataset[:, cat_mask], nan_policy='omit')[0][0]
    expand_mean_n_mode = np.repeat(mean_n_mode[np.newaxis, :], dataset.shape[0], axis=0)
    imputed_dataset[na_mask] = expand_mean_n_mode[na_mask]
    # 2. k-means with mahalanobis distance
    kmeans_est = KMeans(n_clusters=k)
    kmeans_est.fit(imputed_dataset)
    print(kmeans_est.labels)
    # 3. TODO: Impute by Entropy weight method(Only use complete sample to impute)
    complete_idx = np.where(np.sum(na_mask, axis=1)==0)[0]
    tmp_sum=0
    for _cluster_i in range(k): 
        _within_cluster_idx = np.where(kmeans_est.labels==_cluster_i)[0]
        _within_cluster_complete_idx = _within_cluster_idx[np.isin(_within_cluster_idx, complete_idx)]
        _within_cluster_incomplete_idx = _within_cluster_idx[~np.isin(_within_cluster_idx, complete_idx)]
        _attr_max = np.max(dataset[_within_cluster_complete_idx], axis=0)
        tmp_sum+=_within_cluster_incomplete_idx.shape[0]
        # print(tmp_sum)
        def _sim(x, y):
            '''
            similarity between incomplete sample/tuple x, and complete sample y
            TODO: nemerical and categorical attribute need different similarity calculation
            '''
            _na_mask = np.isnan(x)
            _ret_sim = 0
            # numerical distance
            _ret_sim += np.sum( 1 - np.divide(np.abs(x[(~_na_mask)&(~cat_mask)]-y[~_na_mask&(~cat_mask)]), _attr_max[~_na_mask&(~cat_mask)]) )
            # categorical distance
            _ret_sim += np.sum(x[(~_na_mask)&(cat_mask)]!=y[(~_na_mask)&(cat_mask)])
            if np.isnan(_ret_sim): 
                print('An error occur: ', x, ', ', y)
                print(_attr_max)
            return _ret_sim
            
        for _incomplete_idx in _within_cluster_incomplete_idx: 
            _sim_incom_com = np.apply_along_axis(
                lambda x: _sim(dataset[_incomplete_idx], x), 
                1, 
                dataset[_within_cluster_complete_idx]
            )
            _sorted_idx = np.argsort(_sim_incom_com)[::-1]
            _sorted_sim_complete_idx = _within_cluster_complete_idx[_sorted_idx]
            _sorted_sim_complete_sim = _sim_incom_com[_sorted_idx]
            # if number of complete samples in cluster < t, use all of them
            if _within_cluster_complete_idx.shape[0] > t: 
                _sorted_sim_complete_idx = _sorted_sim_complete_idx[:t]
                _sorted_sim_complete_sim = _sorted_sim_complete_sim[:t]
            # print(_sorted_sim_complete_idx)
            
            # Be careful about the samples which is empty and full of nan. 
            # This will make divide 0. 
            # Sol: give all the weight 1. 
            _sorted_sim_complete_weight = None
            if np.sum(_sorted_sim_complete_sim)!=0: 
                _sorted_sim_complete_sim = _sorted_sim_complete_sim / np.sum(_sorted_sim_complete_sim)
                _sorted_sim_complete_entropy = -1 * np.multiply(_sorted_sim_complete_sim, np.log(_sorted_sim_complete_sim))
                _sorted_sim_complete_weight = (1-_sorted_sim_complete_entropy) / (t-np.sum(_sorted_sim_complete_entropy))
            else: 
                _sorted_sim_complete_weight = np.full(_sorted_sim_complete_sim.shape[0], 1/_sorted_sim_complete_sim.shape[0])
            # if np.sum(_sorted_sim_complete_sim)==0: 
            #     _sorted_sim_complete_weight = np.full(_sorted_sim_complete_weight.shape[0],1)
            #     print("sum of sim is 0")
            # print(_sorted_sim_complete_sim)
            # print(_sorted_sim_complete_entropy)
            # print(_sorted_sim_complete_weight)
            # impute numerical
            _incomplete_tuple_incomplete_mask = np.isnan(dataset[_incomplete_idx])
            if np.sum(_incomplete_tuple_incomplete_mask&~cat_mask)>0:
                dataset[_incomplete_idx][_incomplete_tuple_incomplete_mask&~cat_mask] = \
                    np.sum(np.multiply(dataset[_sorted_sim_complete_idx][:, _incomplete_tuple_incomplete_mask&~cat_mask].T, _sorted_sim_complete_weight), axis=1)
            # if np.sum(_incomplete_tuple_incomplete_mask)>0: 
            #     print(_incomplete_tuple_incomplete_mask)
            #     print(dataset[_sorted_sim_complete_idx])
            #     print(np.sum(np.multiply(dataset[_sorted_sim_complete_idx][:, _incomplete_tuple_incomplete_mask&~cat_mask].T, _sorted_sim_complete_weight), axis=1))
            #     print(dataset[_incomplete_idx])
            #     exit()
            # impute categorical
            if np.sum(_incomplete_tuple_incomplete_mask&cat_mask)>0:
                dataset[_incomplete_idx][_incomplete_tuple_incomplete_mask&cat_mask] = \
                    weighted_mode(
                        dataset[_sorted_sim_complete_idx][:, _incomplete_tuple_incomplete_mask&cat_mask].T, 
                        _sorted_sim_complete_weight, 
                        axis=1
                    )[0].T[0]
                
            # exit()
    return dataset

    

if __name__=="__main__": 
    # Get dataset
    full_X, miss_X, miss_mask = load_iris_miss(0.1, 0.5)
    # Generate categorical mask
    # (In iris dataset the last one is categorical)
    cat_mask = np.full(full_X.shape[1], False)
    cat_mask[-1] = True
    
    est = KMeans(n_clusters=3)
    est.fit(full_X[:, :-1])
    # print(est.labels)
    print(full_X[:,-1])

    inputed_dataset = imporved_kmeans_impute(
        3, 
        5, 
        cat_mask, 
        miss_X, 
        lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, cat_mask)
    )
    

    # Evaluation
    cat_mask_expand = np.repeat(cat_mask[np.newaxis, :], full_X.shape[0], axis=0)
    print('MSE: ', mean_squared_error(full_X[miss_mask&~cat_mask_expand], inputed_dataset[miss_mask&~cat_mask_expand]))
    print(full_X[miss_mask&cat_mask_expand])
    print(inputed_dataset[miss_mask&cat_mask_expand])
    print('F1: ', f1_score(full_X[miss_mask&cat_mask_expand], inputed_dataset[miss_mask&cat_mask_expand], average='micro'))


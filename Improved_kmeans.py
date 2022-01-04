import numpy as np
from numpy.core.numeric import full
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from distance_measurement import nan_euclidean_with_categorical
from dataset_prepare import load_iris_miss

'''
Ref: https://github.com/bamtak/machine-learning-implemetation-python/blob/master/KMeans.ipynb
'''
np.random.seed(3)

class KMeans:
    
    def __init__(self, n_clusters=4, max_iter=1e5):
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
        # return np.apply_along_axis(self.compute_label, 1, X)
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

def imporved_kmeans_impute(k, cat_mask, dataset, pairwise_dis_func): 
    # 1. prefill with mean and mode 
    imputed_dataset=dataset.copy()
    # 2. k-means with mahalanobis distance
    # 3. Impute by Entropy weight method
    pass

if __name__=="__main__": 
    # Get dataset
    full_X, miss_X, miss_mask = load_iris_miss(0.2, 0.5)
    # Generate categorical mask
    # (In iris dataset the last one is categorical)
    cat_mask = np.full(full_X.shape[1], False)
    cat_mask[-1] = True

    # x = [[-2.1, -1,  4.3], [3,  1.1,  0.12]]
    # y = [3,  1.1,  0.12]
    # print(np.apply_along_axis(lambda input: np.cov(input,y), 1, x))
    # print(np.linalg.pinv(np.apply_along_axis(lambda input: np.cov(input,y), 1, x)))
    # print(np.linalg.inv(np.apply_along_axis(lambda input: np.cov(input,y), 1, x)[0]))
    # print(np.linalg.pinv(np.apply_along_axis(lambda input: np.cov(input,y), 1, x)[1]))
    est = KMeans(n_clusters=3)
    est.fit(full_X)
    print(est.labels)
    print(full_X[:,-1])
    # imporved_kmeans_impute(
    #     3, 
    #     cat_mask, 
    #     miss_X, 
    #     lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, cat_mask)
    # )

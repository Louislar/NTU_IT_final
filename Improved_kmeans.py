import numpy as np
from numpy.core.numeric import full
from distance_measurement import nan_euclidean_with_categorical
from sklearn.cluster import KMeans
from dataset_prepare import load_iris_miss

'''
Ref: https://github.com/bamtak/machine-learning-implemetation-python/blob/master/KMeans.ipynb
'''
class KMeans:
    
    def __init__(self, n_clusters=4):
        self.K = n_clusters
        
    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        self.intial_centroids = self.centroids
        self.prev_label,  self.labels = None, np.zeros(len(X))
        while not np.all(self.labels == self.prev_label) :
            self.prev_label = self.labels
            self.labels = self.predict(X)
            self.update_centroid(X)
        return self
        
    def predict(self, X):
        return np.apply_along_axis(self.compute_label, 1, X)

    def compute_label(self, x):
        return np.argmin(np.sqrt(np.sum((self.centroids - x)**2, axis=1)))

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

    est = KMeans(n_clusters=3)
    est.fit(full_X)
    print(est.labels)
    print(full_X[:,-1])
    # imporved_kmeans_impute(
    #     3, 
    #     miss_X, 
    #     lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, cat_mask)
    # )

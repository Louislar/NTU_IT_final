import numpy as np
from numpy.core.numeric import full 
import pandas as pd 
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from dataset_prepare import load_iris_miss
from distance_measurement import paired_nan_euclidean_with_categorical


# Get dataset
full_X, miss_X, miss_mask = load_iris_miss(0.1, 0.5)
# Without categrical attribute
# full_X = full_X[:, :-1]
# miss_X = miss_X[:, :-1]
# miss_mask = miss_mask[:, :-1]

# Generate categorical mask
# (In iris dataset the last one is categorical)
cat_mask = np.full(full_X.shape[1], False)
cat_mask[-1] = True


imputer = KNNImputer(
    n_neighbors=5, 
    metric=lambda X,Y,**kwds: paired_nan_euclidean_with_categorical(X,Y,categorical_mask=cat_mask,**kwds)
)
X_pred = imputer.fit_transform(miss_X)
print(X_pred.shape)
print(X_pred[miss_mask].shape)

# Evaluation metric
print('MSE: ', mean_squared_error(full_X[miss_mask], X_pred[miss_mask]))

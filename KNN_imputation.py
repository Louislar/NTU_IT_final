import numpy as np
from numpy.core.numeric import full 
import pandas as pd 
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from dataset_prepare import load_iris_miss
from distance_measurement import paired_nan_euclidean_with_categorical
from distance_measurement import nan_euclidean_with_categorical
from KNN_impute_categorical import KNN_impute_voting


# Get dataset
full_X, miss_X, miss_mask = load_iris_miss(0.1, 0.5)
# Without categrical attribute
# full_X = full_X[:, :-1]
# miss_X = miss_X[:, :-1]
# miss_mask = miss_mask[:, :-1]

# Generate categorical mask
# (In iris dataset the last column is categorical)
cat_mask = np.full(full_X.shape[1], False)
cat_mask[-1] = True

# Numerical imputer(mean)
imputer = KNNImputer(
    n_neighbors=5, 
    metric=lambda X,Y,**kwds: paired_nan_euclidean_with_categorical(X,Y,categorical_mask=cat_mask,**kwds)
)
X_pred = imputer.fit_transform(miss_X)
# print(X_pred)
print('X_pred shape: ', X_pred[miss_mask].shape)

# Categorical imputer(voting)
# Use KNN_impute_categorical.KNN_impute_vote impute the categorical attribute
X_pred_cat = KNN_impute_voting(
    k=5, 
    dataset=miss_X, 
    pairwise_dis_func=lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, cat_mask)
)
# print(X_pred_cat)
print('X_pred_cat shape: ', X_pred_cat.shape)

# Merge numerical and categorical prediction
cat_mask_expand = np.repeat(cat_mask[np.newaxis, :], X_pred.shape[0], axis=0)
X_pred[cat_mask_expand] = X_pred_cat[cat_mask_expand]


# Evaluation metric
## Numerical-RMSE
print('MSE: ', mean_squared_error(full_X[miss_mask&~cat_mask_expand], X_pred[miss_mask&~cat_mask_expand]))
## Categorical-F1
print(full_X[miss_mask&cat_mask_expand])
print(X_pred[miss_mask&cat_mask_expand])
print('F1: ', f1_score(full_X[miss_mask&cat_mask_expand], X_pred[miss_mask&cat_mask_expand], average='micro'))

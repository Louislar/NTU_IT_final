import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataset_prepare import load_iris_miss


'''
RESI framework with sklearn knn imputer, plus customize knn imputer on categorical data
'''
class RESI_imputer: 
    def __init__(self, k, dataset, cat_mask) -> None:
        self.k = k
        self.origin_dataset = dataset
        self.cat_mask = cat_mask
        self.cat_idx = np.where(cat_mask)[0]
        self.na_mask = np.isnan(dataset)
        self.complete_tuple_idx = np.where(np.sum(self.na_mask, axis=1)==0)[0]
        self.incomplete_tuple_idx = np.where(np.sum(self.na_mask, axis=1)!=0)[0]
        # compute attribute weights (numerical and categorical separately on computing parobability)
        # -> Only use complete tuples to compute this
        ## normalize data
        ### numerical
        scaler = MinMaxScaler()
        normalized_numerical_dataset = scaler.fit_transform(dataset[self.complete_tuple_idx][:, ~cat_mask])
        probability_numerical_dataset = normalized_numerical_dataset / np.sum(normalized_numerical_dataset, axis=0)
        ### categorical
        propability_categorical_dataset = dataset[self.complete_tuple_idx][:, self.cat_idx].copy()
        for i in range(self.cat_idx.shape[0]): 
            values, counts = np.unique(propability_categorical_dataset[:, i], return_counts=True)
            probabilities = counts / np.sum(counts)
            for j in range(values.shape[0]): 
                propability_categorical_dataset[:, i][propability_categorical_dataset[:, i]==values[j]] = probabilities[j]
        propability_categorical_dataset = propability_categorical_dataset / np.sum(propability_categorical_dataset, axis=0)
        ### merge
        probability_dataset = np.empty(dataset[self.complete_tuple_idx].shape)
        probability_dataset[:, ~cat_mask] = probability_numerical_dataset
        probability_dataset[:, self.cat_idx] = propability_categorical_dataset
        ## entropy of each attribute
        def _compute_entropy(arr): 
            n = arr.shape[0]
            log_arr = np.log(arr)
            log_arr[np.isneginf(log_arr)] = 0
            entropy = -1 * np.sum(np.multiply(log_arr, arr))
            log_n = np.log(n)
            if np.isneginf(log_n): 
                raise type("error1", (BaseException,), {})(
                "no complete tuples error\n"
                )    # no complete tuples error 
            return entropy / log_n
        entropy_dataset = np.apply_along_axis(_compute_entropy, axis=0, arr=probability_dataset)
        ## weight of each attribute
        weight_attribute = (1 - entropy_dataset) / (entropy_dataset.shape[0] - np.sum(entropy_dataset))
        
    def partition_incompete_tuples(self): 
        '''
        partition incomplete samples by integrity rate
        '''
        self.incomplete_idx_partitions = []
        # compute integrity rate
        pass
    def iteratively_impute_incomplete_tuples(self): 
        iteratively_imputed_dataset = self.origin_dataset.copy()
        pass
    def cross_correct_final_output(self): 
        pass



if __name__=='__main__': 
    # Get dataset
    full_X, miss_X, miss_mask = load_iris_miss(0.1, 0.5)
    # Generate categorical mask
    # (In iris dataset the last one is categorical)
    cat_mask = np.full(full_X.shape[1], False)
    cat_mask[-1] = True

    resi_imputer = RESI_imputer(
        k=5, 
        dataset=miss_X, 
        cat_mask=cat_mask
    )
    pass
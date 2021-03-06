import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from dataset_prepare import load_iris_miss
from distance_measurement import paired_nan_euclidean_with_categorical
from distance_measurement import nan_euclidean_with_categorical
from KNN_impute_categorical import KNN_impute_by_voting_diff_train_test


'''
RESI framework with sklearn knn imputer, plus customize knn imputer on categorical data
'''
class RESI_imputer: 
    def __init__(self, k, m, dataset, cat_mask) -> None:
        '''
        Input: 
        :k: K of KNN
        :m: number of incomplete tuples subset
        '''
        self.k = k
        self.m = m
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
        self.weight_attribute = (1 - entropy_dataset) / (entropy_dataset.shape[0] - np.sum(entropy_dataset))
        
    def partition_incompete_tuples(self): 
        '''
        partition incomplete samples by integrity rate
        '''
        self.incomplete_idx_partitions = [None for i in range(self.m)]
        # compute tuple integrity rate
        tuple_integrity_rate = self.origin_dataset.copy()
        tuple_integrity_rate[self.na_mask]=0
        tuple_integrity_rate[~self.na_mask]=1
        tuple_integrity_rate = np.multiply(tuple_integrity_rate, self.weight_attribute)
        tuple_integrity_rate = np.sum(tuple_integrity_rate, axis=1)
        tuple_integrity_rate = tuple_integrity_rate[self.incomplete_tuple_idx]
        sorted_incomplete_tuple_idx = self.incomplete_tuple_idx[np.argsort(tuple_integrity_rate)[::-1]]
        num_in_subset = np.floor(sorted_incomplete_tuple_idx.shape[0]/self.m).astype(int)
        counter=0
        for i in range(self.m-1): 
            self.incomplete_idx_partitions[i]=\
                sorted_incomplete_tuple_idx[counter:counter+num_in_subset]
            counter += num_in_subset
        self.incomplete_idx_partitions[-1] = \
            sorted_incomplete_tuple_idx[counter:] 

    def iteratively_impute_incomplete_tuples(self): 
        iteratively_complete_idx = self.complete_tuple_idx.copy()
        iteratively_imputed_dataset = self.origin_dataset.copy()
        for _a_incomplete_idx_subset in self.incomplete_idx_partitions: 
            print('incomplete idx: ', _a_incomplete_idx_subset)
            imputer = KNNImputer(
                n_neighbors=self.k, 
                metric=lambda X,Y,**kwds: paired_nan_euclidean_with_categorical(X,Y,categorical_mask=self.cat_mask,**kwds)
            )
            imputer.fit(iteratively_imputed_dataset[iteratively_complete_idx])
            numerical_predict = imputer.transform(iteratively_imputed_dataset[_a_incomplete_idx_subset])
            # print(numerical_predict)
            categorical_predict = KNN_impute_by_voting_diff_train_test(
                k=5, 
                dataset_train=iteratively_imputed_dataset[iteratively_complete_idx], 
                dataset_pred=iteratively_imputed_dataset[_a_incomplete_idx_subset], 
                pairwise_dis_func=lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, self.cat_mask)
            )
            # print(categorical_predict)
            cat_mask_expand = np.repeat(self.cat_mask[np.newaxis, :], iteratively_imputed_dataset[_a_incomplete_idx_subset].shape[0], axis=0)
            numerical_predict[cat_mask_expand] = categorical_predict[cat_mask_expand]
            # fill in predicted tuples
            iteratively_imputed_dataset[_a_incomplete_idx_subset] = numerical_predict
            # complete tuple increases
            iteratively_complete_idx = np.append(iteratively_complete_idx, _a_incomplete_idx_subset)
        self.iteratively_imputed_dataset = iteratively_imputed_dataset
    def cross_correct_final_output(self): 
        '''
        Like k cross fold validation, choose one incomplete subset as validation set, remains as training set
        The last subset only use origin complete subset as training set
        '''
        cross_correction_dataset = self.origin_dataset.copy()
        full_idx = np.arange(cross_correction_dataset.shape[0])
        for _a_incomplete_idx_subset in self.incomplete_idx_partitions[:-1]: 
            print(_a_incomplete_idx_subset)
            train_idx = full_idx[~np.isin(full_idx, _a_incomplete_idx_subset)]
            print(train_idx)
            # numerical
            imputer = KNNImputer(
                n_neighbors=self.k, 
                metric=lambda X,Y,**kwds: paired_nan_euclidean_with_categorical(X,Y,categorical_mask=self.cat_mask,**kwds)
            )
            imputer.fit(self.iteratively_imputed_dataset[train_idx])
            numerical_predict = imputer.transform(cross_correction_dataset[_a_incomplete_idx_subset])
            # categorical
            categorical_predict = KNN_impute_by_voting_diff_train_test(
                k=5, 
                dataset_train=self.iteratively_imputed_dataset[train_idx], 
                dataset_pred=cross_correction_dataset[_a_incomplete_idx_subset], 
                pairwise_dis_func=lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, self.cat_mask)
            )
            # merge
            cat_mask_expand = np.repeat(self.cat_mask[np.newaxis, :], numerical_predict.shape[0], axis=0)
            numerical_predict[cat_mask_expand] = categorical_predict[cat_mask_expand]
            cross_correction_dataset[_a_incomplete_idx_subset] = numerical_predict

        # Last subset only use origin complete subset as training set
        # numerical
        imputer = KNNImputer(
            n_neighbors=self.k, 
            metric=lambda X,Y,**kwds: paired_nan_euclidean_with_categorical(X,Y,categorical_mask=self.cat_mask,**kwds)
        )
        imputer.fit(self.iteratively_imputed_dataset[self.complete_tuple_idx])
        numerical_predict = imputer.transform(cross_correction_dataset[self.incomplete_idx_partitions[-1]])
        # categorical
        categorical_predict = KNN_impute_by_voting_diff_train_test(
            k=5, 
            dataset_train=self.iteratively_imputed_dataset[self.complete_tuple_idx], 
            dataset_pred=cross_correction_dataset[self.incomplete_idx_partitions[-1]], 
            pairwise_dis_func=lambda X,Y: nan_euclidean_with_categorical(X, Y, np.nan, self.cat_mask)
        )
        # merge
        cat_mask_expand = np.repeat(self.cat_mask[np.newaxis, :], numerical_predict.shape[0], axis=0)
        numerical_predict[cat_mask_expand] = categorical_predict[cat_mask_expand]
        cross_correction_dataset[self.incomplete_idx_partitions[-1]] = numerical_predict

        # mean of numerical attributes will be the final output
        cat_mask_expand = np.repeat(self.cat_mask[np.newaxis, :], cross_correction_dataset.shape[0], axis=0)
        cross_correction_dataset[self.na_mask&(~cat_mask_expand)] = \
            cross_correction_dataset[self.na_mask&(~cat_mask_expand)]/2 + self.iteratively_imputed_dataset[self.na_mask&(~cat_mask_expand)]/2
        self.cross_correction_dataset = cross_correction_dataset


def RESI_impute(k, m, miss_data, cat_mask): 
    resi_imputer = RESI_imputer(
        k=k, 
        m=m, 
        dataset=miss_data, 
        cat_mask=cat_mask
    )
    resi_imputer.partition_incompete_tuples()
    resi_imputer.iteratively_impute_incomplete_tuples()
    resi_imputer.cross_correct_final_output()
    return resi_imputer.cross_correction_dataset
    
if __name__=='__main__': 
    # Get dataset
    full_X, miss_X, miss_mask = load_iris_miss(0.1, 0.5)
    # Generate categorical mask
    # (In iris dataset the last one is categorical)
    cat_mask = np.full(full_X.shape[1], False)
    cat_mask[-1] = True

    resi_imputer = RESI_imputer(
        k=5, 
        m=4, 
        dataset=miss_X, 
        cat_mask=cat_mask
    )
    resi_imputer.partition_incompete_tuples()
    resi_imputer.iteratively_impute_incomplete_tuples()
    resi_imputer.cross_correct_final_output()

    # evaluation
    cat_mask_expand = np.repeat(cat_mask[np.newaxis, :], full_X.shape[0], axis=0)
    print('MSE: ', mean_squared_error(full_X[miss_mask&~cat_mask_expand], resi_imputer.cross_correction_dataset[miss_mask&~cat_mask_expand]))
    print(full_X[miss_mask&cat_mask_expand])
    print(resi_imputer.cross_correction_dataset[miss_mask&cat_mask_expand])
    print('F1: ', f1_score(full_X[miss_mask&cat_mask_expand], resi_imputer.cross_correction_dataset[miss_mask&cat_mask_expand], average='micro'))
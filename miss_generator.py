from typing import Counter
import numpy as np

# Given a dataset which have n tuples and s attributes. 
# Given missing rate in float, e.g. 0.01 for 1%. 
# Given complete ratio in float, e.g. 0.5 for 1/2. 

np.random.seed(0)

def gen_rand_nums_given_sum(num_rv, sum): 
    '''
    Special constraint: all generated numbers must greater than 1. 
    Output: 
    (list) a list of random numbers with size=num_rv
    Error: 
    When num_rv > sum, an error occurs
    Ref: 
    https://stackoverflow.com/questions/22380890/generate-n-random-numbers-whose-sum-is-m-and-all-numbers-should-be-greater-than
    '''
    if num_rv>sum:
        return "error"
    cur_sum = num_rv
    rv_arr = np.full(num_rv,1)
    counter=0

    while cur_sum<sum: 
        cur_rand = np.random.choice(cur_sum,1)
        rv_arr[counter]+=cur_rand
        counter+=1
        cur_sum+=cur_rand
        # print(cur_rand)
        # print(rv_arr)
    return rv_arr
    

def generate_missing_value_mask(num_tuples, num_attr, miss_rate, complete_ratio): 
    '''
    Output: 
    (numpy.array) A mask with missing value set as true, not missing values set as false. 
    Error: 
    When number of incomplete tuples greater than number of missing items, an error occurs. 
    '''
    num_incomplete_tuples = np.around(num_tuples*(1-complete_ratio), decimals=0).astype(int)
    num_miss_items = np.around(num_tuples*num_attr*miss_rate).astype(int)
    if num_incomplete_tuples>num_miss_items: 
        return "error"
    miss_mask = np.full((num_tuples, num_attr), False)
    incomplete_tuples_idx = np.random.choice(num_tuples, num_incomplete_tuples, replace=False) # random select without replacement!!
    # TODO: Use gen_rand_nums_given_sum and shuffle the return array, 
    # than apply relevent missing values to the miss_mask
    print(num_incomplete_tuples)
    print(num_miss_items)
    num_miss_items_distribution = gen_rand_nums_given_sum(num_incomplete_tuples, num_miss_items)
    np.random.shuffle(num_miss_items_distribution)    

    for _tuple_idx, _num_miss in zip(incomplete_tuples_idx, num_miss_items_distribution): 
        _miss_attr_idx = np.random.choice(num_attr, _num_miss, replace=False)  # random select without replacement!!
        miss_mask[_tuple_idx, _miss_attr_idx] = True
        print(_tuple_idx, ', ', _num_miss, ', ', _miss_attr_idx)
    print(miss_mask)

if __name__=="__main__": 
    # 10 miss items, 10 complete tupeles
    generate_missing_value_mask(20, 5, 0.1, 0.5)

    # gen_rand_nums_given_sum(5,10)
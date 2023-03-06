import numpy as np
import pickle
import random
random.seed(123456789)

data_dir = 'synthetic_data/'
num_samples = 25
num_folds = 5
fold_size = num_samples // num_folds

perm = np.random.permutation(num_samples)

folds = []
start = 0
while start < num_samples:
    fold = perm[start : start + fold_size]
    folds.append(fold)
    start += fold_size

print(folds)

dictionary = {
    'folds': folds
}
fn = data_dir + '/5_fold_cross_validation.pkl'
f = open(fn, "wb")
pickle.dump(dictionary, f)
f.close()

print('Done')

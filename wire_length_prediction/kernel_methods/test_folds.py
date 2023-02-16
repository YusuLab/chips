import pickle

f = open('4_folds.pkl', 'rb')
dictionary = pickle.load(f)
f.close()

fold_netlists = dictionary['netlists']
fold_indices = dictionary['indices']

num_folds = len(fold_netlists)
print('Number of folds:', num_folds)
assert num_folds == len(fold_indices)

count_netlists = 0
count_samples = 0
for fold in range(num_folds):
    print('Fold', fold, ':')
    print('- Number of netlists:', len(fold_netlists[fold]))
    print('- Number of samples:', len(fold_indices[fold]))
    count_netlists += len(fold_netlists[fold])
    count_samples += len(fold_indices[fold])

print('Total number of netlists:', count_netlists)
print('Total mumber of samples:', count_samples)
print('Done')

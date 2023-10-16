import numpy as np
import pickle
import matplotlib.pyplot as plt

data_dir = 'synthetic_data/'
num_samples = 25

num_instances = []
for sample in range(num_samples):
    fn = data_dir + '/' + str(sample) + '.node_features.pkl'
    f = open(fn, 'rb')
    dictionary = pickle.load(f)
    f.close()
    num_instances.append(dictionary['num_instances'])
num_instances = np.array(num_instances)

print('Minimum size:', np.min(num_instances))
print('Maximum size:', np.max(num_instances))
print('Average size:', np.mean(num_instances))
print('STD size:', np.std(num_instances))

plt.hist(num_instances)
plt.title('Histogram of subgraph sizes')
plt.savefig('histogram_sizes.png')
print('Done sizes')

avg_demand = []
avg_capacity = []
for sample in range(num_samples):
    fn = data_dir + '/' + str(sample) + '.targets.pkl'
    f = open(fn, 'rb')
    dictionary = pickle.load(f)
    f.close()
    demand = np.sum(dictionary['demand'], axis = 1)
    capacity = np.sum(dictionary['capacity'], axis = 1)
    avg_demand.append(np.mean(demand))
    avg_capacity.append(np.mean(capacity))

print('Minimum average demand:', np.min(avg_demand))
print('Maximum average demand:', np.max(avg_demand))
print('Average average demand:', np.mean(avg_demand))
print('STD average demand:', np.std(avg_demand))
print('Done congestion')

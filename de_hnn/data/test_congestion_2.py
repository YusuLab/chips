import numpy as np
import gzip
import json
from scipy.stats import binned_statistic_2d
import time

# Slow getGRCIndex
def getGRCIndex(x, y, xbl, ybl):
    j = 0
    for b in xbl[1:]:
        if x < b:
            break
        j += 1
    i = 0
    for b in ybl[1:]:
        if y < b:
            break
        i += 1
    return i, j

# Fast getGRCIndex
def binary_search(key, array):
    if key < array[0]:
        return 0
    start = 0
    finish = len(array) - 1
    best = 0
    while start <= finish:
        mid = (start + finish) // 2
        if key >= array[mid]:
            best = max(best, mid)
            start = mid + 1
        else:
            finish = mid - 1
    return best + 1

def getGRCIndex(x, y, xbl, ybl):
    j = binary_search(x, xbl[1:])
    i = binary_search(y, ybl[1:])
    return i, j

# Design
# design = 'counter'
# design = 'xbar'
design = 'RocketTile'
print('Design:', design)

# Information about instances and nets
instances_nets_fn = 'NCSU-DigIC-GraphData-2022-10-15/' + design + '/' + design + '.json.gz'

with gzip.open(instances_nets_fn, 'r') as fin:
    instances_nets_data = json.load(fin)

instances = instances_nets_data['instances']
nets = instances_nets_data['nets']

num_instances = len(instances)
num_nets = len(nets)

print('Number of instances:', num_instances)
print('Number of nets:', num_nets)

# Congestion
congestion_fn = 'NCSU-DigIC-GraphData-2022-10-15/' + design + '/' + design + '_congestion.npz'

# Load congestion data file
congestion_data = np.load(congestion_fn)

congestion_data_demand = congestion_data['demand']
congestion_data_capacity = congestion_data['capacity']

num_layers = len(list(congestion_data['layerList']))
print('Number of layers:', num_layers)
print('Layers:', list(congestion_data['layerList']))

for layer in list(congestion_data['layerList']):
    print(layer, '--------------------------')
    lyr = list(congestion_data['layerList']).index(layer)
    ybl = congestion_data['yBoundaryList']
    xbl = congestion_data['xBoundaryList']

    # Get placements info
    t = time.time()
    xloc_list = [instances[idx]['xloc'] for idx in range(num_instances)]
    yloc_list = [instances[idx]['yloc'] for idx in range(num_instances)]
    print('Time for getting placement info:', time.time() - t)

    # Binned statistics 2D
    t = time.time()
    ret = binned_statistic_2d(xloc_list, yloc_list, None, 'count', bins = [xbl[1:], ybl[1:]], expand_binnumbers = True)
    print('Time for binned statistics:', time.time() - t)

    i_list = np.array([ret.binnumber[0, idx] - 1 for idx in range(num_instances)])
    j_list = np.array([ret.binnumber[1, idx] - 1 for idx in range(num_instances)])

    # Get demand and capacity
    t = time.time()
    demand_list = congestion_data_demand[lyr, i_list, j_list].flatten()
    capacity_list = congestion_data_capacity[lyr, i_list, j_list].flatten()
    print('Time to get demand and capacity:', time.time() - t)

    demand_list = np.array(demand_list)
    capacity_list = np.array(capacity_list)

    average_demand = np.mean(demand_list)
    average_capacity = np.mean(capacity_list)
    average_diff = np.mean(capacity_list - demand_list)
    count_congestions = np.sum(demand_list > capacity_list)
 
    print()
    print('Number of demand > capacity:', count_congestions)
    print('Average capacity - demand:', average_diff)
    print('Average demand:', average_demand)
    print('Average capacity:', average_capacity)

print('Done')

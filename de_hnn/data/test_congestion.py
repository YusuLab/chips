import numpy as np
import gzip
import json

# Slow getGRCIndex
'''
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
'''

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
design = 'xbar'
# design = 'RocketTile'
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

num_layers = len(list(congestion_data['layerList']))
print('Number of layers:', num_layers)
print('Layers:', list(congestion_data['layerList']))

for layer in list(congestion_data['layerList']):
    print(layer, '--------------------------')
    lyr = list(congestion_data['layerList']).index(layer)
    ybl = congestion_data['yBoundaryList']
    xbl = congestion_data['xBoundaryList']

    count_congestions = 0
    sum_diff = 0
    sum_demand = 0
    sum_capacity = 0

    for idx in range(num_instances):
        xloc = instances[idx]['xloc']
        yloc = instances[idx]['yloc']
        i, j = getGRCIndex(xloc, yloc, xbl, ybl)

        demand = congestion_data['demand'][lyr][i][j]
        capacity = congestion_data['capacity'][lyr][i][j]

        # Statistics
        sum_diff += (capacity - demand)
        sum_demand += demand
        sum_capacity += capacity

        # If demand is bigger than capacity then increse the number of congestions by 1
        if demand > capacity:
            count_congestions += 1

        if (idx + 1) % 1000 == 0:
            print('Done scanning for', idx + 1, 'instances')

    average_diff = sum_diff / num_instances
    average_demand = sum_demand / num_instances
    average_capacity = sum_capacity / num_instances
    
    print('Number of demand > capacity:', count_congestions)
    print('Average capacity - demand:', average_diff)
    print('Average demand:', average_demand)
    print('Average capacity:', average_capacity)
print('Done')

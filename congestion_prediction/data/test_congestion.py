import numpy as np
import gzip
import json

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

# Design
# design = 'counter'
design = 'xbar'
# design = 'RocketTile'

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

lyr = list(congestion_data['layerList']).index('M1')
ybl = congestion_data['yBoundaryList']
xbl = congestion_data['xBoundaryList']

count_congestions = 0
for idx in range(num_instances):
    xloc = instances[idx]['xloc']
    yloc = instances[idx]['yloc']
    i, j = getGRCIndex(xloc, yloc, xbl, ybl)

    demand = congestion_data['demand'][lyr][i][j]
    capacity = congestion_data['capacity'][lyr][i][j]

    # If demand is bigger than capacity then increse the number of congestions by 1
    if demand > capacity:
        count_congestions += 1

    if (idx + 1) % 1000 == 0:
        print('Done scanning for', idx + 1, 'instances')

print('Number of congestions:', count_congestions)
print('Done')

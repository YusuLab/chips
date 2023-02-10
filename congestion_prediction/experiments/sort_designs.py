import numpy as np
import json
import gzip

data_dir = '../data/RosettaStone-GraphData-2023-01-21/'

all_designs = [
    'adaptec1',
    'adaptec2',
    'adaptec3',
    'adaptec4',
    'adaptec5',
    'bigblue1',
    'bigblue2',
    'bigblue3',
    'bigblue4',
    'newblue1',
    'newblue2',
    'newblue3',
    'newblue4',
    'newblue5',
    'newblue6',
    'newblue7',
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue6',
    'superblue7',
    'superblue9',
    'superblue10',
    'superblue11',
    'superblue12',
    'superblue14',
    'superblue15',
    'superblue16',
    'superblue18',
    'superblue19'
]

num_designs = len(all_designs)
print('Number of designs:', num_designs)

all_sizes = []
has_placement = []
for design in all_designs:
    conn_fn = data_dir + '/' + design + '/' + design + '_connectivity.npz'
    json_fn = data_dir + '/' + design + '/' + design + '.json.gz'
    
    # Connectivity
    print('Reading', conn_fn)
    conn = np.load(conn_fn)
    row = conn['row']
    col = conn['col']
    data = conn['data']
    shape = conn['shape']

    num_instances = max(row) + 1
    all_sizes.append(num_instances)

    # Placement
    with gzip.open(json_fn, 'r') as fin:
        data = json.load(fin)

    instances = data['instances']
    nets = data['nets']

    if instances[0]['xloc'] != 0 or instances[0]['yloc'] != 0:
        has_placement.append(design)

for i in range(num_designs):
    for j in range(i + 1, num_designs):
        if all_sizes[i] > all_sizes[j]:
            temp = all_sizes[i]
            all_sizes[i] = all_sizes[j]
            all_sizes[j] = temp

            temp = all_designs[i]
            all_designs[i] = all_designs[j]
            all_designs[j] = temp

print('Sort designs by size:')
for i in range(num_designs):
    print(all_designs[i], all_sizes[i])

print('List of designs with placements:', has_placement)

print('Done')

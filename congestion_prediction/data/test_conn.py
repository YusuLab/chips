import numpy as np

# Old data
file_name = 'NCSU-DigIC-GraphData-2022-10-15/counter/counter_connectivity.npz'

# RosettaStone
# file_name = 'RosettaStone-GraphData-2023-01-21/adaptec1/adaptec1_connectivity.npz'

# Load data file
data = np.load(file_name)

print(data)
print(data.files)

print('row')
print(data['row'])
print(data['row'].shape)

print('col')
print(data['col'])
print(data['col'].shape)

print('data')
print(data['data'])
print(data['data'].shape)

print(min(data['data']))
print(max(data['data']))

print('shape')
print(data['shape'])
print(data['shape'].shape)

print('Done')

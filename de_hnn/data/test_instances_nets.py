import numpy as np
import gzip
import json

# Old data
file_name = 'NCSU-DigIC-GraphData-2022-10-15/counter/counter.json.gz'

# RosettaStone
# file_name = 'RosettaStone-GraphData-2023-01-21/adaptec1/adaptec1.json.gz'

with gzip.open(file_name, 'r') as fin:
    data = json.load(fin)

print(len(data))
print(data.keys())

instances = data['instances']
nets = data['nets']

print(len(instances))
print(len(nets))

print(instances[0])
print(instances[1])

print(nets[0])
print(nets[1])

print('Done')

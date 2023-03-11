import numpy as np
import gzip
import json

file_name = 'RosettaStone-GraphData-2023-01-21/cells.json.gz'

with gzip.open(file_name, 'r') as fin:
    data = json.load(fin)

print(len(data))
print(data[0])
print(data[1])
print(data[2])

print('Done')

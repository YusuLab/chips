import numpy as np
import gzip

file_name = 'RosettaStone-GraphData-2023-01-21/adaptec1/adaptec1.json.gz'

with gzip.open(file_name, 'r') as fin:
    for line in fin:
        print('got line', line)

print('Done')

import numpy as np
import gzip
import json
import matplotlib.pyplot as plt

file_name = 'RosettaStone-GraphData-2023-03-06/cells.json.gz'

with gzip.open(file_name, 'r') as fin:
    cell_data = json.load(fin)

widths = []
heights = []
for idx in range(len(cell_data)):
    width = cell_data[idx]['width']
    height = cell_data[idx]['height']
    widths.append(width)
    heights.append(height)

widths = np.array(widths)
heights = np.array(heights)

print('min width:', np.min(widths))
print('max width:', np.max(widths))
print('mean width:', np.mean(widths))
print('std width:', np.std(widths))
print()
print('min height:', np.min(heights))
print('max height:', np.max(heights))
print('mean height:', np.mean(heights))
print('std height:', np.std(heights))

widths = (widths - np.min(widths)) / (np.max(widths) - np.min(widths))
heights = (heights - np.min(heights)) / (np.max(heights) - np.min(heights))

print('------------------------------------')
print('min width:', np.min(widths))
print('max width:', np.max(widths))
print('mean width:', np.mean(widths))
print('std width:', np.std(widths))
print()
print('min height:', np.min(heights))
print('max height:', np.max(heights))
print('mean height:', np.mean(heights))
print('std height:', np.std(heights))

fig, axs = plt.subplots(2)
axs[0].hist(widths, density = True, bins = 100)  # density = False would make counts
axs[0].set_title('Widths')
axs[1].hist(heights, density = True, bins = 100)  # density = False would make counts
axs[1].set_title('Heights')
fig.suptitle('Histogram of cell sizes')
plt.savefig('histogram_cells.png')

print('Done')

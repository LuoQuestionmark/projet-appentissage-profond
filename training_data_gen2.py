"""
This program generate the array data for further training using
the midi files in the './data' dir.
"""

from mid_processing import DataGenerator1, MidProcess
from matplotlib import pyplot as plt

import os
import numpy as np

# data = list()
# for root, dirs, files in os.walk("./data"):
#     for filename in files:
#         file_path = os.path.join(root, filename)

#         print(f"parsing file {filename}")
#         mp = MidProcess(file_path)
#         data.extend(mp.parse())

mp = MidProcess('./data/Fugue3.mid')
test_data = mp.parse()

data = np.array([int(i.end_t - i.start_t) for i in test_data if i.channel==1])

print(data)

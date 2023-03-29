"""
This program generate the array data for further training using
the midi files in the './data' dir.
"""

from mid_processing import DataGenerator1, MidProcess

import os
import numpy as np

data = list()
for root, dirs, files in os.walk("./data"):
    for filename in files:
        file_path = os.path.join(root, filename)

        print(f"parsing file {filename}")
        mp = MidProcess(file_path)
        data.extend(mp.parse())

mp = MidProcess('./data/Fugue1.mid')
test_data = mp.parse()

dg = DataGenerator1()
train, label = dg.generate(data)
test, t_label = dg.generate(test_data)


np.save('./train_data/train_data', train)
np.save('./train_data/train_label', label)
np.save('./train_data/test_data', test)
np.save('./train_data/test_label', t_label)

"""
This program generate the array data for further training using
the midi files in the './data' dir.
"""

from typing import Iterable
from mid_processing import MidProcess
# from matplotlib import pyplot as plt

import os
import numpy as np

# data = list()
# for root, dirs, files in os.walk("./data"):
#     for filename in files:
#         file_path = os.path.join(root, filename)

#         print(f"parsing file {filename}")
#         mp = MidProcess(file_path)
#         data.extend(mp.parse())

def downsize(val):
    return round(val / 20)

class DataGenerator2:
    """
    Output will be used as the input of a LSTM network, which asks the data in the shape
    of [timesteps, batch, feature].
    For the current version, feature=1 (length of a note);
    and the timesteps=20, as each one is just a fragment of music score
    """
    def __init__(self, timesteps=20) -> None:
        self.timesteps = timesteps

    def generate(self, data: Iterable[int]):
        batch = len(data) // self.timesteps
        out = np.zeros((self.timesteps, batch, 1), dtype=np.uint8)
        for b in range(batch):
            out[:, b, 0] = data[b * self.timesteps:(b+1) * self.timesteps]

        return out

# mp = MidProcess('./data/Fugue3.mid')
# test_data = mp.parse()

# data = np.array([int(i.end_t - i.start_t) for i in test_data if i.channel==1])

# print([downsize(i) for i in data])

if __name__ == '__main__':
    data = list()
    dg = DataGenerator2()
    for root, dirs, files in os.walk("./data"):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"parsing file {filename}")
            mp = MidProcess(file_path)
            local_data = mp.parse()
            for chan in range(0, 6):
                chan_data = [round((i.end_t - i.start_t) / 20) for i in local_data if i.channel==chan]
                if len(chan_data) == 0: continue
                data.append(dg.generate(chan_data))

                # double data by seperate differently the data
                data.append(dg.generate(chan_data[10:]))

    training_data = np.concatenate(data, axis=1, dtype=float)

    np.save('./train_data/train_serial_data', training_data)

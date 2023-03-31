"""
This program reads the "model_1" created by the "model_1.py", loads
with the trained data.
"""

from model_1 import create_model
from itertools import count, compress
from collections import deque

import numpy as np
import tensorflow as tf

# create the model and load the data from checkpoint
model = create_model()
model.load_weights("./checkpoints/model_1")

# test the model
# test_data   = np.load("./train_data/test_data.npy")
# test_label  = np.load("./train_data/test_label.npy")
# model.evaluate(test_data, test_label, verbose=2)

# test the power of this network:
# start by play g - c - d - e
notes = deque([55, 60, 62, 64])
input_array = np.zeros((1, 128))
for i in notes:
    input_array[0][i] = 1

# make the system predict what are the most possible notes
prediction = model.predict(input_array)[0]

for iteration in range(10):
    # select the most possible new note
    prediction = prediction ** 2
    possibilities = prediction / sum(prediction)

    new_note = np.random.choice(128, p=possibilities)
    # and get its index

    # drop a note from the current table
    drop_index = notes.popleft()
    notes.append(new_note)

    # create new input
    input_array = np.zeros((1, 128))
    for i in notes:
        input_array[0][i] = 1

    # each time the drop note is the output
    print(f"drop: {drop_index}")

    # loop, feed the new result into the model
    prediction = model.predict(input_array)[0]

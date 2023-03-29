"""
This program reads the "model_1" created by the "model_1.py", loads
with the trained data.
"""

from model_1 import create_model
import numpy as np
import tensorflow as tf

model = create_model()
model.load_weights("./checkpoints/model_1")

test_data   = np.load("./train_data/test_data.npy")
test_label  = np.load("./train_data/test_label.npy")

model.evaluate(test_data, test_label, verbose=2)

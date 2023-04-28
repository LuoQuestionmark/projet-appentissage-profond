from random import randint

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

data = np.load("./train_data/train_serial_data.npy")
data = data.transpose(1, 0, 2)
data /= 128

# input format:
# (batch_size, 20, 1)
# 20 is the manually selected number as the size of a single segment
# for the training, we will use the first 15 as input, the last 5 as output

train_data  = data[10:,:15,:]
train_label = data[10:,15:,:]

test_data  = data[:10,:15,:]
test_label = data[:10,15:,:]

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=15, output_dim=5))
    model.add(tf.keras.layers.LSTM(10))
    model.add(tf.keras.layers.Dense(5))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss='binary_crossentropy'
        # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

model = create_model()

model.fit(train_data, train_label, validation_data=(test_data, test_label))

model.save_weights('./checkpoints/model_2')

# noise = np.random.rand(1, 20, 1)
# prediction = model.predict(noise)

# print(noise)
# print([round(i * 128) for i in prediction[0]])

# motive = [[4,4,2,2,2,2,1,1,1,1,1,1,4,4,4]]
# prediction = model.predict(motive)
# print(prediction)
# print([round(i * 128) for i in prediction[0]])

# outputs = model (data)
# outputs = [round(i) for i in outputs]
# print(outputs)

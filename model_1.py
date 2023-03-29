import os
import numpy as np
import tensorflow as tf

from mid_processing import DataGenerator1, MidProcess

data = list()
for root, dirs, files in os.walk("./data"):
    for filename in files:
        file_path = os.path.join(root, filename)

        print(f"parsing file {filename}")
        mp = MidProcess(file_path)
        data.extend(mp.parse())
    print("flag")


# mp = MidProcess('Fugue1.mid')
# out = mp.parse()
dg = DataGenerator1()
train, label = dg.generate(data)
test, t_label = dg.generate(mp.parse())

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape=(128,)))

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss='binary_crossentropy'
    # metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(train, label, validation_data=(test, t_label))

import numpy as np
import tensorflow as tf

from mid_processing import DataGenerator1, MidProcess

mp = MidProcess('Fugue1.mid')
out = mp.parse()
dg = DataGenerator1()
train, label = dg.generate(out)
test, t_label = dg.generate(out)

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

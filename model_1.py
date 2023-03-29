import numpy as np
import tensorflow as tf

train_data  = np.load("./train_data/train_data.npy")
train_label = np.load("./train_data/train_label.npy")
test_data   = np.load("./train_data/test_data.npy")
test_label  = np.load("./train_data/test_label.npy")

def create_model():
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

    return model

if __name__ == '__main__':
    model = create_model()

    model.fit(train_data, train_label, validation_data=(test_data, test_label))

    model.save_weights('./checkpoints/model_1')

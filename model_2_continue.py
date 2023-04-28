import tensorflow as tf
import numpy as np

from model_2 import create_model

model = create_model()
model.load_weights("./checkpoints/model_2")

noise = np.random.rand(1, 20, 1)
prediction = model.predict(noise)

print(noise)
print([round(i * 128) for i in prediction[0]])

motive = [[4,4,2,2,2,2,1,1,1,1,4,4,4]]
prediction = model.predict(motive)
print(prediction)
print([round(i * 128) for i in prediction[0]])

pass

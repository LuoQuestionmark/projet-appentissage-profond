"""
This program combine the two model created seperately
"""

import tensorflow as tf
import numpy as np
import random

from model_1 import create_model as model_1_init
from model_2 import create_model as model_2_init


class CombinedModel:
    model_1 = model_1_init()
    model_2 = model_2_init()
    model_1.load_weights("./checkpoints/model_1")
    model_2.load_weights("./checkpoints/model_2")

    def __init__(self) -> None:
        pass

    def generate_voice_subject_duration(self, length=10, iter_num=10):
        """
        This function generate a voice subject from random noise,
        after multiple iterations it should generate something in the right style
        """
        out = np.zeros((1, 20 + 5 * iter_num, 1))
        input_val = np.random.rand(1, 20, 1)
        out[:, 0:20, :] = input_val
        for i in range(iter_num):
            prediction = CombinedModel.model_2.predict(input_val).reshape(1,5,1)
            out[:, 20 + 5 * i:25 + 5 * i, :] = prediction
            input_val = out[:, 5 + 5 * i:25 + 5 * i, :]

        out = out[:, 20+5*iter_num-length:,:]
        out = np.clip(out, a_min=0, a_max=None)
        return out
    
    def develop_voice_subject_duration(self, subject, sub_length=10, dev_length=100, iter_num=100):
        """
        This function develop the subject input by copying the subject and fill the gap with the trained model
        """
        out = np.zeros((1, dev_length, 1))

        actions = random.choices(population=["copy", "fill"], weights=[0.3, 0.7], k=iter_num)
        for action in actions:
            if action == "copy":
                index = random.randint(0, dev_length - sub_length)
                out[:,index:index+sub_length,:] = subject
            elif action == "fill":
                index = random.randint(0, dev_length - 20 - 5)
                model_input = out[:,index:index+20,:]
                model_output = CombinedModel.model_2.predict(model_input)
                out[:,index+20:index+25,:] = np.clip(model_output.reshape(1, 5, 1), a_min=0, a_max=None)
            else:
                pass

        return out

if __name__ == '__main__':
    cm = CombinedModel()
    subject = cm.generate_voice_subject_duration()
    develop = cm.develop_voice_subject_duration(subject)

    print(develop)

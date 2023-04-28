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
        This function generate a voice subject (duration) from random noise,
        after multiple iterations it should generate something in the right style
        """
        out = np.random.rand(1, 20 + 5 * iter_num, 1)
        input_val = np.random.rand(1, 20, 1)
        out[:, 0:20, :] = input_val
        for i in range(iter_num):
            prediction = CombinedModel.model_2.predict(input_val).reshape(1,5,1)
            out[:, 20 + 5 * i:25 + 5 * i, :] = np.abs(prediction)
            input_val = out[:, 5 + 5 * i:25 + 5 * i, :]

        out = out[:, 20+5*iter_num-length:,:]
        out = np.clip(out, a_min=0, a_max=None)
        return out
    
    def develop_voice_subject_duration(self, subject, sub_length=10, dev_length=100, iter_num=300):
        """
        This function develop the subject input by copying the subject and fill the gap with the trained model
        """
        out = np.zeros((1, dev_length, 1))

        actions = random.choices(population=["copy", "fill", "noise"], weights=[0.3, 0.69, 0.01], k=iter_num)
        for action in actions:
            if action == "copy":
                index = random.randint(0, dev_length - sub_length - 1)
                copy_times = 1
                """
                add variantion: augmentation of subject
                """
                while random.random() < 0.3:
                    copy_times *= 2
                out[:,index:index+sub_length,:] = subject * copy_times
            elif action == "fill":
                index = random.randint(0, dev_length - 20 - 5 - 1)
                model_input = out[:,index:index+20,:]
                model_output = CombinedModel.model_2.predict(model_input)
                out[:,index+20:index+25,:] = np.clip(model_output.reshape(1, 5, 1), a_min=0, a_max=None)
            elif action == "noise":
                index = random.randint(0, dev_length - 1)
                out[0,index,0] = random.random()
            else:
                raise RuntimeError("unexpected case")

        return out
    
    def postprocess_voice_subject_duration(self, develop):
        develop = develop.flatten()
        develop *= 256
        develop = np.log2(develop).astype(int)
        develop = np.clip(develop, a_min=0, a_max=4)
        develop = np.power(2, develop)
        return np.clip(develop.astype(int), a_min=0, a_max=16)

    
    def generate_voice_subject_pitch(self, length=10, iter_num=10):
        """
        This function generate a voice subject (pitch) from random noise,
        after multiple iterations it should generate something in the right style
        """
        input_val = np.random.rand(1, 128)
        for _ in range(iter_num):
            output_val = CombinedModel.model_1.predict(input_val).reshape(1, 128)
            input_val = output_val
        # output_val is now the probability of Bach's style
        output_val = output_val / np.sum(output_val, axis=None)
        
        # now try to generate the subject
        # start by a mostly random one
        out = random.choices(range(128), weights=output_val[0], k=length)

        # replace the notes with the model prediction
        for index in random.choices(range(length), k=iter_num):
            input_val = np.zeros((1, 128))
            # fill all the neighborhood values
            for j in range(max(0, index - 5), min(length, index + 5)):
                input_val[0, out[j]] = 1
            # but not the index one
            input_val[0, index] = 0

            output_val = CombinedModel.model_1.predict(input_val).reshape(1, 128)
            output_val = output_val / np.sum(output_val, axis=None)
            out[index] = np.random.choice(128, p=output_val[0])

        return out
    
    def develop_voice_subject_pitch(self, subject, sub_length=10, dev_length=100, iter_num=100, decl=0):
        out = np.zeros((1, dev_length), dtype=int)

        # prepare the subject
        subject = np.array(subject)
        subject = subject.reshape(1, sub_length) + decl
        np.clip(subject, a_min=0, a_max=127, out=subject)
        out[:, 0:sub_length] = subject

        actions = random.choices(population=["copy", "fill", "noise"], weights=[0.3, 0.65, 0.05], k=iter_num)
        for action in actions:
            if action == "copy":
                index = random.randint(0, dev_length - sub_length - 1)
                out[:,index:index+sub_length] = subject
            elif action == "fill":
                index = random.randint(0, dev_length - 1)
                input_val = np.zeros((1, 128))
                # fill all the neighborhood values
                for j in range(max(0, index - 5), min(dev_length, index + 5)):
                    input_val[0, out[0,j]] = 1
                # but not the index one
                input_val[0, index] = 0
                model_output = CombinedModel.model_1.predict(input_val)
                prob = model_output[0] / np.sum(model_output, axis=None)
                out[:,index] = np.random.choice(128, p=prob)
            elif action == "noise":
                index = random.randint(0, dev_length - 1)
                out[0,index] = np.random.normal(64, 8)
            else:
                raise RuntimeError("unexpected case")
        return out
    
    def postprocess_voice_subject_pitch(self, develop):
        develop = develop.flatten()
        out = np.zeros_like(develop)
        mean = np.mean(develop)
        for i, e in enumerate(develop):
            while abs(e - mean) > 12:
                if e > mean:
                    e -= 12
                else:
                    e += 12
            out[i] = e

        return out


if __name__ == '__main__':
    cm = CombinedModel()
    # subject_duration = cm.generate_voice_subject_duration()
    # develop_duration = cm.develop_voice_subject_duration(subject_duration)
    # result_duration = cm.postprocess_voice_subject_duration(develop_duration)

    # print(subject_duration)
    # print(develop_duration)
    # print(result_duration)

    subject = cm.generate_voice_subject_pitch()
    develop = cm.develop_voice_subject_pitch(subject)
    post = cm.postprocess_voice_subject_pitch(develop)
    print(develop)
    print(post)

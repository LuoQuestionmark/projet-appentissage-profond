"""
this file is the prototype of the midi file processing
later I will create a class to handle it in a more standard way.
"""

from mido import MidiFile

import numpy as np
import random

# mid = MidiFile('Fugue1.mid')

# # the follwing lines is the example from the library "mido"
# for i, track in enumerate(mid.tracks):
#     print('Track {}: {}'.format(i, track.name))
#     for msg in track:
#         if not msg.is_meta:
#             print(msg)

# noticing that the output is like
# note_on ... note_off with a start time and a finish time, a pitch, etc.

# note_on channel=1 note=60 velocity=64 time=0
# note_off channel=1 note=60 velocity=51 time=60
# note_on channel=1 note=62 velocity=64 time=0
# note_off channel=1 note=62 velocity=49 time=60

# note_on / off: as the name suggest
# note: pitch
# velocity: strength, doesn't matter
# time: past time since the last msg

class Note:
    """
    The class for each and every sing note
    """
    def __init__(self, start_t, end_t, pitch) -> None:
        self.start_t = start_t
        self.end_t = end_t
        self.pitch = pitch

        self.neighbors = set()

    def __str__(self) -> str:
        return f"Note {self.pitch}: {self.start_t}-{self.end_t} with {len(self.neighbors)} neighbors"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.end_t - self.start_t

    def get_group(self) -> set:
        out = set([i.pitch for i in self.neighbors])
        out.add(self.pitch)

        return out 
        

class MidProcess:
    def __init__(self, file) -> None:
        try:
            self.mid_file = MidiFile(file)
        except OSError:
            self.mid_file = None
            print(f"error in file {file}")

    def parse(self) -> list[Note]:
        """
        The output will be presented as a list of Note
        """
        out = list()

        if self.mid_file is None:
            return out

        tmp_dict = dict()
        # firstly, create a temperal list that save the information of uncompleted note,
        # i.e. a "note_on" on a certain pitch p
        # the key will be the pitch p and the value will be the start time t
        
        # first step: parse through all the track, all the notes
        # this will iterate through all the track one by one
        for track in self.mid_file.tracks:
            # restart the time count
            timestamp = 0
            for msg in track:
                # ignore the meta-info
                if msg.is_meta:
                    continue
                
                timestamp += msg.time

                # consider various situations
                if msg.type == 'note_on':
                    tmp_dict[msg.note] = timestamp
                elif msg.type == 'note_off':
                    t_s = tmp_dict[msg.note]
                    t_e = timestamp
                    pitch = msg.note

                    out.append(Note(t_s, t_e, pitch))
                else:
                    print(msg.type)
                    exit(0)

        # second step: update the nearest neighbors, within
        # a maximum distance of three times average note_length
        # it should be done with a O(log(n)) algo,
        # but I will just do it with O(n^2), knowing that the
        # input size should not matter too much in our case
        average_len = np.mean([len(n) for n in out])
        dist_max = 2.5 * average_len
        out = sorted(out, key=lambda i: i.start_t)
        for index, note in enumerate(out):
            for note2 in out[index:]:
                if np.abs(note.end_t - note2.start_t) < dist_max:
                    note.neighbors.add(note2)
                    note2.neighbors.add(note)

        return out

class DataGenerator1:
    def __init__(self, hide_num=1, input_size=128, output_size=128) -> None:
        self.hide_num = hide_num
        self.input_size = input_size
        self.output_size = output_size


    def generate(self, data) -> tuple[np.array, np.array]:
        """
        First attempt of data generation based on the data parsed by MidProcess class

        output: (data, label)
        """
        out_train = np.empty((len(data), self.input_size))

        out_label = np.empty((len(data), self.output_size))

        for i, note in enumerate(data):
            notes = note.get_group()
            label = set(random.choices(list(notes), k=self.hide_num))
            train = notes - label

            train_array = np.zeros(self.input_size, dtype=np.int0)
            for j in train:
                train_array[j] = 1

            label_array = np.zeros(self.output_size, dtype=np.int0)
            for j in label:
                label_array[j] = 1

            # TODO: add random noise if needed

            out_train[i] = train_array
            out_label[i] = label_array

        return out_train, out_label



if __name__ == "__main__":
    mp = MidProcess('Fugue1.mid')
    out = mp.parse()
    # print(out)

    # average_len = np.mean([len(n) for n in out])
    # print(f"average length: {average_len}")
    
    dg = DataGenerator1()
    train, label = dg.generate(out)

    print(np.sum(train, axis=1))

"""
this file is the prototype of the midi file processing
later I will create a class to handle it in a more standard way.
"""

from mido import MidiFile

mid = MidiFile('Fugue1.mid')

# the follwing lines is the example from the library "mido"
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)
# noticing that the output is like
# note_on ... note_off with a start time and a finish time, a pitch, etc.

# note_on channel=1 note=60 velocity=64 time=0
# note_off channel=1 note=60 velocity=51 time=60
# note_on channel=1 note=62 velocity=64 time=0
# note_off channel=1 note=62 velocity=49 time=60
# note_on channel=1 note=60 velocity=64 time=0
# note_off channel=1 note=60 velocity=55 time=60
# note_on channel=1 note=59 velocity=64 time=0
# note_off channel=1 note=59 velocity=59 time=60
# note_on channel=1 note=57 velocity=64 time=0

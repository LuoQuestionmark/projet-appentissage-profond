"""
This file contains a class that can translate the midi pitch (from 0 to 127)
to the natural notation (C#, F, G#, etc.)
"""

class MidTranslator:
    letter_table = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    @staticmethod
    def translate(pitch: int) -> str:
        letter = MidTranslator.letter_table[pitch // 12]
        number = str(pitch % 12)

        return letter + number

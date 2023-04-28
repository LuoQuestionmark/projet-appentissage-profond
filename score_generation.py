from model_combined import CombinedModel

class ScoreGeneration:
    def __init__(self, vocal=3) -> None:
        cm = CombinedModel()
        self.results_duration = list()
        self.results_pitch = list()
        
        subject_pitch = cm.generate_voice_subject_pitch()
        subject_duration = cm.generate_voice_subject_duration()
        
        for i in range(vocal):
            develop_duration = cm.develop_voice_subject_duration(subject_duration)
            result_duration = cm.postprocess_voice_subject_duration(develop_duration)

            self.results_duration.append(result_duration)

            if i == 0:
                decl = 0
            elif i == 1:
                decl = 7
            elif i == 2:
                decl = -12
            elif i == 3:
                decl = -5
            elif i == 4:
                decl = 12
            else:
                decl = -1
                raise RuntimeError("do not take this into consideration, too many vocals")

            develop_pitch = cm.develop_voice_subject_pitch(subject_pitch, decl=decl)
            result_pitch = cm.postprocess_voice_subject_pitch(develop_pitch)

            self.results_pitch.append(result_pitch)

    def coordinate(self):
        """
        need a function to coordinate the relationship between vocals 
        """
        pass

    def note_to_lilypond(self, p, d):
        letter_table = ["c", "cis", "d", "dis", "e", "eis", "f", "fis", "g", "gis", "a", "ais", "b"]

        letter = letter_table[p % 12]

        number = p // 12
        while abs(number != 4):
            if number > 4:
                letter += "'"
                number -= 1
            else:
                letter += ","
                number += 1

        if d == 1:
            letter += "8"
        elif d == 2:
            pass
        elif d == 4:
            letter += "2"
        else:
            letter += "1"
        
        return letter



    def __str__(self) -> str:
        out = "\\version \"2.20.0\"\n"
        out += "{\n"

        for pitches, durations in zip(self.results_pitch, self.results_duration):
            for p, d in zip(pitches, durations):
                out += self.note_to_lilypond(p, d)
                out += " "
        out += "}"

        return out
    
if __name__ == '__main__':
    sg = ScoreGeneration(1)
    print(sg)
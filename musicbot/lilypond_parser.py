from musicbot import models


class LilyPondParser():
    """ class to write a partition from a list of SimpleModel notes. """
    def __init__(self, note_sequence):
        self.notes = [models.SimpleModel.from_array(note) for note in note_sequence]

    def write_score(self, score="test_score.ly"):
        with open(score, 'w') as file:
            file.write('\\version "2.16.2"\n')
            file.write("\score{\n")
            file.write(self.get_score())
            file.write("\\layout { }\n")
            file.write("\\midi { }\n")
            file.write('}\n')

    def get_score(self):
        voices = 0
        res = ""
        res += "<< \n "
        res += "\\new Staff{\n"
        res += '\\set Staff.midiInstrument = "grand"\n'
        res += "\\new Voice {\n"
        start = True
        brack_open = False

        for i, n in enumerate(self.notes):
            if n.ts == 0:
                if voices > 0:
                    res += " } \n"
                else:
                    res += "<< \n "
                res += "\n \\new Voice {\n"
                voices = 1
            else:
                if voices > 0:
                    res += " } \n"
                    res += " >> \n"
                    voices = 0
            if n.ts != 0:
                if not start:
                    res += ">> \n "
                    res += self.notes[i - 1].get_rest_time(n.ts)
                res += "<< \n "
                brack_open = True
                start = False
            res += n.__str__()

        if voices > 0:
            res += " } \n"
            res += " >> \n"
    
        if brack_open:
            res += ">> \n "
        res += "}\n"
        res += "}\n"
        res += ">> \n "
        
        return res


def main():
    n_sequence = 200
    notes = [models.random_arr() for i in range(n_sequence)]
    LilyPondParser(notes).write_score()


def gamme():
    """[ts, td, dot, o, tone]"""
    notes = [
        [2, 2, 0, 0,  3, 0],
        [2, 2, 0, 0, 3, 2],
        [2, 2, 0, 0, 3, 4],
        [2, 2, 0, 0, 3, 5],
        [1, 2, 0, 0, 3, 7],
        [2, 2, 0, 0, 3, 9],
        [2, 2, 0, 0, 3, 11],
        [2, 2, 0, 0, 4, 0]
    ]
    LilyPondParser(notes).write_score()


if __name__ == "__main__":
    #main()
    gamme()

            




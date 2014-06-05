""" model representation of a music score : """


note_name = {
    0: 'c',
    1: 'cis',
    2: 'd',
    3: 'dis',
    4: 'e',
    5: 'f',
    6: 'fis',
    7: 'g',
    8: 'gis',
    9: 'a',
    10: 'ais',
    11: 'b',
}


class SimpleModel:
    """ in the simple model, a music note has 4 parameters :
        - ts : the time since the last note.
        - td : the duration of the note (0 whole note, 1 half, 2 quarter)
        - dot : add a dot to the duration.
        - r : (boolean : true -> rest, false note)
        - o : the octave (int within [0, 6]) -> 7 octaves
        - tone: (int within [0, 11]) """

    def __init__(self, ts, td, dot, o, tone):
        self.ts = ts
        self.td = td
        self.dot = dot
        self.o = o
        self.tone = tone

    def get_octave_string(self):
        res = ''
        if 6 >= self.o >= 3:
            for i in range(2, self.o):
                res += "'"
        elif 0 <= self.o <= 6:
            for i in range(self.o, 3):
                res += ","
        else:
            print "Warning wrong number of octave"
        return res

    def get_duration_string(self):
        res = ""
        if 0 <= self.td <= 7:
            res += "{}".format(2 ** self.td)
        else:
            print " Warning wrong duration {}".format(self.td)
        if self.dot:
            res += "."
        return res 
    
    def __str__(self):
        """ return the lilypond format of the note 
            see http://www.lilypond.org/doc/v2.18/Documentation/learning/simple-notation"""
        res = ''
        if self.r:
            res += 'r'
        else:
            res += note_name[self.tone]
        res += self.get_octave_string()
        res += self.get_octave_string()
        return res

            

        

        




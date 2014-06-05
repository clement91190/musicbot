""" model representation of a music score : """

import numpy as np
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
        - ts : the time since the last note . [0 : 8]
        - td : the duration of the note (0 whole note, 1 half, 2 quarter) [0: 8]
        - dot : add a dot to the duration. [0 or 1]
        - o : the octave (int within [0, 6]) -> 7 octaves
        - r : True -> special note (rest)
        - tone: (int within [0, 11]) """

    def __init__(self, ts, td, dot, r, o, tone):
        self.ts = ts
        self.td = td
        self.dot = dot
        self.r = r
        self.o = o
        self.tone = tone

    def get_octave_string(self):
        res = ''
        if 6 >= self.o >= 3:
            for i in range(2, self.o):
                res += "\'"
        elif 0 <= self.o <= 2:
            for i in range(self.o, 3):
                res += ","
        else:
            print "Warning wrong number of octave"
        return res

    def get_duration_string(self):
        res = ""
        if 0 <= self.td <= 8:
            res += "{}".format(2 ** self.td)
        else:
            print " Warning wrong duration {}".format(self.td)
        if self.dot:
            res += "."
        return res 

    def get_rest_time(self, ts):
        if ts == self.td:
            return ""
        elif ts < self.td:
            res = ""
            for t in range(self.td, ts, -1):
                res += " r{} ".format(2 ** t)
            res += "\n"
            return res
        else:
            print "Warning, timing too complicated, skipping to end of note"
            return ""
    
    def __str__(self):
        """ return the lilypond format of the note 
            see http://www.lilypond.org/doc/v2.18/Documentation/learning/simple-notation"""
        res = ''
        if self.r:
            res += 'r'
        else:
            res += note_name[self.tone]
            res += self.get_octave_string()
            res += self.get_duration_string()
        return res

    @staticmethod
    def from_array(arr):
        """ from a numpy array, we create the note """
        return SimpleModel(*list(arr))


def random_arr():
    ts = np.random.randint(2, 5)
    td = np.random.randint(2, 5)
    dot = np.random.randint(0, 2)
    o = np.random.randint(3, 5)
    r = 0
    #r = np.random.randint(0, 2)
    tone = np.random.randint(0, 12)
    return [ts, td, dot, r, o, tone]


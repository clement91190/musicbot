import numpy
import cPickle
#from data.bouncing_ball.bouncing_ball import bounce_vec, kl_seq
import glob
import theano
import random
import theano.tensor as T
from RNN.theano_hf.hf import hf_optimizer, SequenceDataset
from midi.utils import midiread, midiwrite
import pylab


class MusicBrain:
    def __init__(self):
        self.r = (21, 109)
        self.seq_length = 10
        self.dt = 0.3
        self.show = True
        self.p = None
        self.load_dataset()

    def load_dataset(self):
        re = ["data/JSB Chorales/train/*.mid"]
        re.append("data/JSB Chorales/test/*.mid")
        re.append("data/JSB Chorales/valid/*.mid")
        files = []
        for r in re:
            files += glob.glob(r)
        assert(len(files) > 0)
        print "generating dataset..."
        dataset = [midiread(f, self.r, self.dt).piano_roll.astype(theano.config.floatX) for f in files]
    
        memorization_dataset = [[]]  # memorize the first unit for 100 time-steps with binary noise

        n = 0

        for seq in dataset:
            for i in range(0, len(seq), self.seq_length):
                memorization_dataset[0].append(seq[i:i + self.seq_length])

        while False: #n < 100000:
            sequence = random.choice(dataset)
            if len(sequence) < self.seq_length:
                print " to short !"
            i = random.choice(range(0, len(sequence), self.seq_length))
            memorization_dataset[0].append(sequence[i:i + self.seq_length])
            n = n + 1

        print "number of sequence for training : ", len(memorization_dataset[0])

        self.train = [memorization_dataset[0][:-1000]]
        self.valid = [memorization_dataset[0][-1000:]]

        self.gradient_dataset = SequenceDataset(self.train, batch_size=None, number_batches=1000)
        self.cg_dataset = SequenceDataset(self.train, batch_size=None, number_batches=500)
        self.valid_dataset = SequenceDataset(self.valid, batch_size=None, number_batches=500)

    def train_classical_music(self):

        self.simple_RNN(100, self.r[1] - self.r[0])

        hf_optimizer(self.p, self.inputs, self.s, self.costs, 0.5*(self.h + 1), self.ha).train(self.gradient_dataset, self.cg_dataset, initial_lambda=0.5, mu=1.0, preconditioner=False, validation=self.valid_dataset, plot_cost_file="plot_cost_music.pkl", num_updates=2000, save_callback=self.save)

# single-layer recurrent neural network with sigmoid output, only last time-step output is significant
    def simple_RNN(self, nh, n_in, load_file=None):
        if load_file is None:
            Wx = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_in, nh)).astype(theano.config.floatX))
            Wh = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
            Wy = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nh, n_in)).astype(theano.config.floatX))
            bh = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
            by = theano.shared(numpy.zeros(n_in, dtype=theano.config.floatX))
            h0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
            p = [Wx, Wh, Wy, bh, by, h0]
        else:
            with open(load_file, 'r') as f:
                p = [Wx, Wh, Wy, bh, by, h0] = cPickle.load(f)
        x = T.matrix()

        def recurrence(x_t, h_tm1):
            ha_t = T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh
            h_t = T.tanh(ha_t)
            s_t = T.dot(h_t, Wy) + by
            return [ha_t, h_t, s_t]

        ([ha, h, activations], updates) = theano.scan(fn=recurrence, sequences=x[:-1], outputs_info=[dict(), h0, dict()])

        h = T.tanh(ha)  # so it is differentiable with respect to ha
        t = x[-1, :]
        s = activations[-1, :]
        y = T.nnet.sigmoid(s)
        loss = -t*T.log(y + 1e-14) - (1-t)*T.log((1-y) + 1e-14)
        acc = T.neq(T.round(y), t)
        self.generate_function = theano.function([x], y)
        self.p, self.inputs,  self.s, self.costs, self.h, self.ha = p, [x], s, [T.mean(loss), T.mean(acc)], h, ha

    def generate_sample(self, length=200, filename="sample.mid"):
        piece = self.valid[0][2]
        piece.shape
        for i in range(length - self.seq_length):
            seq = numpy.array(piece[-self.seq_length:])
            #print seq.shape
            new = self.generate_function(seq)
            sample = numpy.random.random(88)
            new_n = new > sample
            #while numpy.sum(new_n) == 0:
            #    new += 0.001
            #    new_n = new > sample
            
            #print new
            piece = numpy.vstack([piece, new_n])
            #print piece.shape
        piano_roll = piece
        midiwrite(filename, piano_roll, self.r, self.dt)
        if self.show:
            extent = (0, self.dt * len(piano_roll)) + self.r
            pylab.figure()
            pylab.imshow(
                piano_roll.T, origin='lower', aspect='auto',
                interpolation='nearest', cmap=pylab.cm.gray_r,
                extent=extent)
            pylab.xlabel('time (s)')
            pylab.ylabel('MIDI note number')
            pylab.title('generated piano-roll')
            pylab.savefig('piano_roll')
            pylab.show()

    def save(self, filename="musicbrain.pkl"):
        with open(filename, 'w') as file:
            cPickle.dump(self.p, file)


def main():
    m_brain = MusicBrain()
    try:
        m_brain.train_classical_music()
    except KeyboardInterrupt:
        print 'Interrupted by user.'
    m_brain.generate_sample()

    
if __name__ == "__main__":
    main()


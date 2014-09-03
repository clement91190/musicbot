import theano
import theano.tensor as T
import numpy as np


def evolution_step(p, mask, validation_set):
    sgrnn = SGRNNStep(p, mask, validation_set)
    return sgrnn.run()


class SGRNNStep:
    def __init__(self, p, mask, validation_set):
        self.p = [self.Wx, self.whh, self.Wy, self.bh, self.by, self.h0] = p
        self.mask = mask
        print " shape of the mask : ", self.mask.shape
        self.validation_set = validation_set
        self.whh_copy = np.array(self.whh, dtype=theano.config.floatX)
        self.rnn = SimpleRNN(p, mask)

    def preliminary_calculation(self):
        scores = []
        for i in range(self.whh.shape[0]):
            print i
            self.brain_deletion([i])
            self.rnn.set_whh(self.whh)
            scores.append(self.rnn.get_score(self.validation_set))
        self.scores = np.array(scores)
        self.scores = self.scores - np.min(self.scores)
        self.whh = self.whh_copy
        print self.scores
            
    def brain_deletion(self, l):   
        self.whh = np.array(self.whh_copy)
        for i in l:
#delete row
            self.whh[i] = 0
#delete column
            self.whh[:, i] = 0

    def selection(self, k=5):
        ind = sorted(range(self.whh.shape[0]), key=lambda i: self.scores[i])[:-k]
        self.whh = self.whh[ind][:, ind]
        self.mask = self.mask[ind][:, ind]

        self.Wx = self.Wx[:, ind]
        self.Wy = self.Wy[ind]
        self.bh = self.bh[ind]
        self.h0 = self.h0[ind]

        print self.mask.shape, self.whh.shape



    def cross_over(self, k=10):
        ind = sorted(range(self.whh.shape[0]), key=lambda i: self.scores[i])[:k]
        for i in ind:
            self.insert_in_model(i)

    def insert_in_model(self, i):
        #insert in whh
        self.whh = np.array(np.hstack([self.whh, np.matrix(self.whh[:, i]).T]))  # TODO noise
        self.whh[i] = 0.5 * self.whh[i]  # TODO add noise
        self.whh = np.vstack([self.whh, self.whh[i]])  # TODO add noise

        self.mask = np.vstack([self.mask, self.mask[i]])
        self.mask = np.array(np.hstack([self.mask, np.matrix(self.mask[:, i]).T]))

        self.Wx = np.array(np.hstack([self.Wx, np.matrix(self.Wx[:, i]).T])) 
        self.bh = np.hstack([self.bh, self.bh[i]]) 
        self.h0 = np.hstack([self.h0, self.h0[i]])

        self.Wy[:, i] = 0.5 * self.Wy[:, i]  
        self.Wy = np.array(np.vstack([self.Wy, self.Wy[i]])) 

        self.p = [self.Wx, self.whh, self.Wy, self.bh, self.by, self.h0]

    def run(self):
        self.preliminary_calculation()
        self.selection()
        self.cross_over()
        return self.p, self.mask


class SimpleRNN:
    def __init__(self, p_init, mask):
        [Wx, Wh, Wy, bh, by, h0] = p_init
        for i in p_init:
                print i.shape, i.dtype

        print mask.shape, mask.dtype
        #raw_input()

        Wx = theano.shared(Wx).astype(theano.config.floatX)
        self.Wh = theano.shared(Wh).astype(theano.config.floatX)
        Wy = theano.shared(Wy).astype(theano.config.floatX)
        bh = theano.shared(bh).astype(theano.config.floatX)
        by = theano.shared(by).astype(theano.config.floatX)
        h0 = theano.shared(h0).astype(theano.config.floatX)
        p = [Wx, Wh, Wy, bh, by, h0]
        x = T.matrix()

        def recurrence(x_t, h_tm1):
            ha_t = T.dot(x_t, Wx) + T.dot(h_tm1, Wh * mask) + bh
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
        self.f_cost = theano.function([x], T.mean(loss), on_unused_input='ignore')  # for quick cost evaluation
        self.p, self.inputs,  self.s, self.costs, self.h, self.ha = p, [x], s, [T.mean(loss), T.mean(acc)], h, ha
    
    def set_whh(self, whh):
        self.Wh.set_value(whh)

    def get_score(self, valid_dataset):
        res = np.mean([self.f_cost(*i) for i in valid_dataset.iterate()], axis=0)
        return res



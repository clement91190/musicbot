import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator

import time
import os
import datetime
import cPickle as pickle
from theano_hf.hf import hf_optimizer, SequenceDataset
from rnn import RNN


import matplotlib.pyplot as plt
plt.ion()

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'


class RNNHfOptim(BaseEstimator):
    def __init__(self, n_in=5, n_hidden=50, n_out=5, 
                 L1_reg=0.00, L2_reg=0.00,
                 activation='tanh', output_type='real',
                 use_symbolic_softmax=False):
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.use_symbolic_softmax = use_symbolic_softmax
        
        self.ready()
        self.tune_optimizer()

    def tune_optimizer(
            self, initial_lambda=0.1, mu=0.03, global_backtracking=False,
            preconditioner=False, max_cg_iterations=250,
            num_updates=5, validation=None, validation_frequency=1,
            patience=np.inf, save_progress=None, cg_number_batches=20, 
            gd_number_batches=100):
        #TODO write all parameters with descriptions

        self.initial_lambda = initial_lambda
        self.mu = mu
        self.global_backtracking = global_backtracking
        self.preconditioner = preconditioner
        self.max_cg_iterations = max_cg_iterations
        self.n_updates = num_updates
        self.validation = validation
        self.validation_frequency = validation_frequency
        self.patience = patience
        self.save_progress = save_progress
        self.cg_number_batches = cg_number_batches
        self.gd_number_batches = gd_number_batches


    def ready(self):
        # input (where first dimension is time)
        self.x = T.matrix()
        # target (where first dimension is time)
        if self.output_type == 'real':
            self.y = T.matrix(name='y', dtype=theano.config.floatX)
        elif self.output_type == 'binary':
            self.y = T.matrix(name='y', dtype='int32')
        elif self.output_type == 'softmax':  # only vector labels supported
            self.y = T.vector(name='y', dtype='int32')
        else:
            raise NotImplementedError
        # initial hidden state of the RNN
        self.h0 = T.vector()
        # learning rate
        self.lr = T.scalar()

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        self.rnn = RNN(
            input=self.x, n_in=self.n_in,
            n_hidden=self.n_hidden, n_out=self.n_out,
            activation=activation, output_type=self.output_type,
            use_symbolic_softmax=self.use_symbolic_softmax)

        if self.output_type == 'real':
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.rnn.y_pred,
                                           mode=mode)
        elif self.output_type == 'binary':
            self.predict_proba = theano.function(
                inputs=[self.x, ], outputs=self.rnn.p_y_given_x, mode=mode)
            self.predict = theano.function(
                inputs=[self.x, ],
                outputs=T.round(self.rnn.p_y_given_x),
                mode=mode)
        elif self.output_type == 'softmax':
            self.predict_proba = theano.function(
                inputs=[self.x, ],
                outputs=self.rnn.p_y_given_x, mode=mode)
            self.predict = theano.function(
                inputs=[self.x, ],
                outputs=self.rnn.y_out, mode=mode)
        else:
            raise NotImplementedError

    def shared_dataset(self, data_xy):
        """ Load the dataset into shared variables """

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX))

        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX))

        if self.output_type in ('binary', 'softmax'):
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y

    def __getstate__(self):
        """ Return state sequence."""
        params = self._get_params()  # parameters set in constructor
        weights = [p.get_value() for p in self.rnn.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.rnn.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights = state
        self.set_params(**params)
        self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

    def fit(self, X_train, Y_train):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        X_train : ndarray (n_seq x n_steps x n_in)
        Y_train : ndarray (n_seq x n_steps x n_out)

        """
        # SequenceDataset wants a list of sequences
        # this allows them to be different lengths, but here they're not
        seq = [i for i in X_train]
        targets = [i for i in Y_train]

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print "building model..."

        #TODO : batch_size in parameters.
        gradient_dataset = SequenceDataset(
            [seq, targets], batch_size=None, number_batches=self.gd_number_batches)
        cg_dataset = SequenceDataset(
            [seq, targets], batch_size=None, number_batches=self.cg_number_batches)

        cost = self.rnn.loss(self.y) \
            + self.L1_reg * self.rnn.L1 \
            + self.L2_reg * self.rnn.L2_sqr

        opt = hf_optimizer(
            p=self.rnn.params, inputs=[self.x, self.y],
            s=self.rnn.y_pred,
            costs=[cost], h=self.rnn.h)

        ###############
        # TRAIN MODEL #
        ###############
        print "starting training ..."

        opt.train(gradient_dataset, cg_dataset, num_updates=self.n_updates)


def test_real():
    """ Test RNN with real-valued outputs. """
    n_hidden = 10
    n_in = 3
    n_out = 3
    n_steps = 100
    n_seq = 10

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    seq= np.asfarray(seq, dtype='float32')
    targets = np.zeros((n_seq, n_steps, n_out), dtype='float32')

    targets[:, 1:, 0] = seq[:, :-1, 2]  # delayed 1
    targets[:, 1:, 1] = seq[:, :-1, 1]  # delayed 1
    targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

    targets += 0.01 * np.random.standard_normal(targets.shape)

    model = RNNHfOptim(n_in=n_in, n_hidden=n_hidden, n_out=n_out, activation='tanh')

    model.tune_optimizer(num_updates=100)
    model.fit(seq, targets)

    plt.close('all')
    #fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq[0])
    ax1.set_title('input')

    ax2 = plt.subplot(212)
    true_targets = plt.plot(targets[0])

    guess = model.predict(seq[0])
    guessed_targets = plt.plot(guess, linestyle='--')
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')
    plt.show()


if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    test_real()
    # problem takes more epochs to solve
    #test_binary(multiple_out=True, n_epochs=2400)
    #test_softmax(n_epochs=250)
    print "Elapsed time: %f" % (time.time() - t0)
    raw_input()

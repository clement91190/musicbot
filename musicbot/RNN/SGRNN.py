import numpy as np
import random
import theano
import theano.tensor as T
import pygraphviz as pgv
from sklearn.base import BaseEstimator
import networkx as nx
from rnn import RNN


class WeightsHandler():
    def __init__(self, n_in, n_out, satu_weights = 100, n_hidden_start=10):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden_start
        self.satu_weights = satu_weights

        #weights and biases
        self.W_in = np.asarray(np.random.uniform(
            size=(self.n_in, self.n_hidden), low=-.01, high=.01),
            dtype=theano.config.floatX)
        self.W = np.asarray(np.random.uniform(
            size=(self.n_hidden, self.n_hidden), low=-.01, high=.01),
            dtype=theano.config.floatX)
        self.W_out = np.asarray(np.random.uniform(
            size=(self.n_hidden, self.n_out), low=-.01, high=.01),
            dtype=theano.config.floatX)

        #changing W -> keeping only k-neighbors connections.

        k = 3
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if abs(i - j) > k:
                    self.W[i, j] = 0.

        self.bh = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.h0 = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.by = np.zeros((self.n_out,), dtype=theano.config.floatX)

        #gradient matrix for hidden_weights. 
        self.grad_W = np.zeros((self.n_hidden, self.n_hidden), dtype=theano.config.floatX)

    def clean_small_weights(self, eps=0.00001):
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if self.grad_W[i, j] ** 2 + self.W[i, j] ** 2 < eps:
                    self.W[i, j] = 0.

    def add_weights(self, eps=0.5):

        random_range = 0.1
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if i >= j:
                    ind_i, ind_j = j, i
                else:
                    ind_i, ind_j = i, j

                if ind_j > ind_i and ind_i < self.n_hidden - 1 and ind_j > 0:
                    if self.W[ind_i, ind_j - 1] > eps and self.W[ind_i + 1, ind_j] > eps:
                        self.W[ind_i, ind_j] = (random.random() * 2 - 1) * random_range

    def add_complex_node(self, cluster_treshold=0.4, std_noise=0.01):
        """ duplicate the node with the highest clustering coefficient(with a small noise)"""

        self.Ghh = nx.Graph(np.array(self.W, dtype=[('weight', float)]))
        clustering_coefs = nx.algorithms.cluster.clustering(self.Ghh, weight="weight")
        ind, val = max(clustering_coefs.items(), key=lambda (i, j): j)

        if val > 0.4:
            self.n_hidden += 1
            new_W = np.zeros((self.n_hidden, self.n_hidden), dtype=theano.config.floatX)
        #TODO add Win Wout
        upper_ind_i = False
        for i in range(self.n_hidden):
            upper_ind_j = False
            if i == ind or i == ind + 1:
                upper_ind_i = True
                for j in range(self.n_hidden):
                    if j == ind or j == ind + 1:
                        upper_ind_j = True
                        val = self.W[ind, ind] 
                        new_W[i, j] = np.random.normal(val/4, std_noise * val)
                    else:
                        val = self.W[ind, j - 1 * upper_ind_j]
                        new_W[i, j] = np.random.normal(val/2, std_noise * val)
            else:
                for j in range(self.n_hidden):
                    if j == ind or j == ind + 1:
                        upper_ind_j = True
                        val = self.W[i - 1 * upper_ind_i, ind]
                        new_W[i, j] = np.random.normal(val/2, std_noise * val)
                    else:
                        new_W[i, j] = self.W[i - 1 * upper_ind_i, j - 1 * upper_ind_j]

    def reorganize(self):
        #TODO
        pass

    def evolution(self):
        self.clean_small_weights()
        self.add_weights()
        self.add_complex_node()

    def define_mask(self):
        self.mask = self.W != 0

    def define_v_W(self):
        self.location = []
        self.v_W = []
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                val = self.W[i, j]
                if val != 0:
                    self.location.append([i, j])
                    self.v_W.append(val)

        self.location = np.asarray(self.location, dtype="int32")
        self.v_W = np.asarray(self.v_W, dtype="float32")

    def to_theano_params(self):
        """ return a tuple (list of parameters (for W -> vector of non-zero value), list of theano shared variables: [W_in, W, W, bh, h0, hy] ) """

        #self.define_v_W()
        self.define_mask()
        W_in = theano.shared(value=self.W_in, name='W_in')
        W_out = theano.shared(value=self.W_out, name='W_out')
        W = theano.shared(value=self.W, name='W')
        h0 = theano.shared(value=self.h0, name='h0')
        bh = theano.shared(value=self.bh, name='bh')
        by = theano.shared(value=self.by, name='by')

        """
        values = theano.shared(value=self.v_W, name='v_W')
        location = theano.shared(value=self.location, name='location')
        output_model = theano.shared(value=np.zeros((self.n_hidden, self.n_hidden), dtype="float32"), name='outputmodel')
        
        def set_value_at_position(a_location, a_value, output_model):
            zeros_subtensor = output_model[a_location[0], a_location[1]]
            return T.set_subtensor(zeros_subtensor, a_value)

        outputs_info = T.zeros_like(output_model, dtype="float32")
        result, updates = theano.scan(
            fn=set_value_at_position,
            outputs_info=outputs_info,
            sequences=[location, values])

        W = result[-1]
        """
        #params = [W_in, W_out, values, h0, bh, by]
        weights = [W_in, W_out, W, h0, bh, by]

        return (self.mask, weights) 


class SGRNN(RNN):
    """ redefine the constructor of RNN to take a WeightsHandler as parameter. """
    def __init__(self, input, weight_handler, activation=T.tanh,
                 output_type='real', use_symbolic_softmax=False):

        self.input = input
        self.activation = activation
        self.output_type = output_type

        # when using HF, SoftmaxGrad.grad is not implemented
        # use a symbolic softmax which is slightly slower than T.nnet.softmax
        # See: http://groups.google.com/group/theano-dev/browse_thread/
        # thread/3930bd5a6a67d27a
        if use_symbolic_softmax:
            def symbolic_softmax(x):
                e = T.exp(x)
                return e / T.sum(e, axis=1).dimshuffle(0, 'x')
            self.softmax = symbolic_softmax
        else:
            self.softmax = T.nnet.softmax

        self.mask, self.params = weight_handler.to_theano_params()
        self.W_in, self.W_out, self.W, self.h0, self.bh, self.by = self.params
        self.init_structure()
        

class GraphDraw(BaseEstimator):
    def __init__(self, n_in=5, n_out=5, activation='tanh', output_type='real'):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.output_type = output_type

    def set_weights(self, W, W_in, W_out, bh, bi):
        self.W = W
        self.W_in = W_in
        self.W_out = W_out
        self.bh = bh
        self.bi = bi

    def mutation(self):
        """ this is where the magic happends"""
        pass

    def get_weights(self):
        return (self.W, self.W_in, self.W_out, self.bh, self.bi)


    def draw(self, file='neural_net.png'):
        """ represent the graph of neurons using graphviz """
        graph = pgv.AGraph(directed=False, strict=True, overlap=False)
        #draw input to hidden
        for inc_in, input, in enumerate(self.W_in):
            for inc_hidden,  weight in enumerate(input):
                if weight != 0:
                    graph.add_edge("in{}".format(inc_in), "h{}".format(inc_hidden), color="blue")  # label=t,) arrowhead='diamond')
                    node = graph.get_node("in{}".format(inc_in))
                    node.attr['style'] = 'filled'
                    node.attr['fillcolor'] = 'blue'
                    node.attr['shape'] = 'circle'
                    node.attr['label'] = ''

        #draw hidden to hidden
        for inc_hidden1, hidden_row in enumerate(self.W):
            for inc_hidden2, weight in enumerate(hidden_row):
                if weight != 0:
                    graph.add_edge("h{}".format(inc_hidden1), "h{}".format(inc_hidden2), color="red")  # label=t,) arrowhead='diamond')
                    node = graph.get_node("h{}".format(inc_hidden1))
                    node.attr['style'] = 'filled'
                    node.attr['fillcolor'] = 'red'
                    node.attr['shape'] = 'circle'
                    node.attr['label'] = ''

        #draw hidden to output
        for inc_hidden, hidden in enumerate(self.W_out):
            for inc_out, weight in enumerate(hidden_row):
                if weight != 0:
                    graph.add_edge("h{}".format(inc_hidden), "o{}".format(inc_out), color="green")  # label=t,) arrowhead='diamond')
                    node = graph.get_node("o{}".format(inc_out))
                    node.attr['style'] = 'filled'
                    node.attr['fillcolor'] = 'green'
                    node.attr['shape'] = 'circle'
                    node.attr['label'] = ''
        graph.layout(prog='neato')
        graph.draw(file)
        import os
        os.system("xdg-open {}".format(file))


def main():
    n_in = 4
    n_hidden = 16
    n_out = 4

    W_in = np.ones((n_in, n_hidden))
    W = np.random.randint(2, size=(n_hidden, n_hidden))
    #W = np.identity(n_hidden)
    W_out = np.ones((n_hidden, n_out))
    bh = None
    bi = None

    nn = SGRNN()
    nn.set_weights(W, W_in, W_out, bh, bi)
    nn.draw()


if __name__ == "__main__":
    main()

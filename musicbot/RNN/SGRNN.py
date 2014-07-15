import numpy as np
import random
import theano
import theano.tensor as T
import pygraphviz as pgv
from sklearn.base import BaseEstimator
import networkx as nx
from rnn import RNN


class WeightsHandler():
    def __init__(self, n_in, n_out, satu_weights, n_hidden_start=10):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden_start
        self.satu_weights = satu_weights

        #weights and biases
        self.Wih = np.asarray(np.random.uniform(
            size=(self.n_in, self.n_hidden), low=-.01, high=.01),
            dtype=theano.config.floatX)
        self.Whh = np.asarray(np.random.uniform(
            size=(self.n_hidden, self.n_hidden), low=-.01, high=.01),
            dtype=theano.config.floatX)
        self.Who = np.asarray(np.random.uniform(
            size=(self.n_hidden, self.n_out), low=-.01, high=.01),
            dtype=theano.config.floatX)

        #changing Whh -> keeping only k-neighbors connections.

        k = 3
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if abs(i - j) > k:
                    self.Whh[i, j] = 0.

        self.bh = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
        self.ho = np.zeros((self._hidden,), dtype=theano.config.floatX)
        self.by = np.zeros((self.n_out,), dtype=theano.config.floatX)

        #gradient matrix for hidden_weights. 
        self.grad_Whh = np.zeros((self.n_hidden, self.n_hidden), dtype=theano.config.floatX)

    def clean_small_weights(self, eps=0.00001):
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if self.grad_Whh[i, j] ** 2 + self.Whh[i, j] ** 2 < eps:
                    self.Whh[i, j] = 0.

    def add_weights(self, eps=0.5):

        random_range = 0.1
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                if i >= j:
                    ind_i, ind_j = j, i
                else:
                    ind_i, ind_j = i, j

                if ind_j > ind_i and ind_i < self.n_hidden - 1 and ind_j > 0:
                    if self.Whh[ind_i, ind_j - 1] > eps and self.Whh[ind_i + 1, ind_j] > eps:
                        self.Whh[ind_i, ind_j] = (random.random() * 2 - 1) * random_range

    def add_complex_node(self, cluster_treshold=0.4, std_noise=0.01):
        """ duplicate the node with the highest clustering coefficient(with a small noise)"""

        self.Ghh = nx.Graph(np.array(self.Whh, dtype=[('weight', float)]))
        clustering_coefs = nx.algorithms.cluster.clustering(self.Ghh, weight="weight")
        ind, val = max(clustering_coefs.items(), key=lambda (i, j): j)

        if val > 0.4:
            self.n_hidden += 1
            new_Whh = np.zeros((self.n_hidden, self.n_hidden), dtype=theano.config.floatX)

        upper_ind_i = False
        for i in range(self.n_hidden):
            upper_ind_j = False
            if i == ind or i == ind + 1:
                upper_ind_i = True
                for j in range(self.n_hidden):
                    if j == ind or j == ind + 1:
                        upper_ind_j = True
                        val = self.Whh[ind, ind] 
                        new_Whh[i, j] = np.random.normal(val/4, std_noise * val)
                    else:
                        val = self.Whh[ind, j - 1 * upper_ind_j]
                        new_Whh[i, j] = np.random.normal(val/2, std_noise * val)
            else:
                for j in range(self.n_hidden):
                    if j == ind or j == ind + 1:
                        upper_ind_j = True
                        val = self.Whh[i - 1 * upper_ind_i, ind]
                        new_Whh[i, j] = np.random.normal(val/2, std_noise * val)
                    else:
                        new_Whh[i, j] = self.Whh[i - 1 * upper_ind_i, j - 1 * upper_ind_j]

    def reorganize(self):
        #TODO
        pass

    def evolution(self):
        self.clean_small_weights()
        self.add_weights()
        self.add_complex_node()

    def define_v_Whh(self):
        self.location = []
        self.v_Whh = []
        for i in range(self.n_hidden):
            for j in range(self.n_hidden):
                val = self.Whh[i, j]
                if val != 0:
                    self.location.append(i, j)
                    self.v_Whh.append(val)

    def to_theano_params(self):
        """ return a tuple (list of parameters (for Whh -> vector of non-zero value), list of theano shared variables: [Wih, Whh, Whh, bh, ho, hy] ) """
        W_in = theano.shared(value=self.W_in, name='W_in')
        W_out = theano.shared(value=self.W_out, name='W_out')
        h0 = theano.shared(value=self.h0, name='h0')
        bh = theano.shared(value=self.bh, name='bh')
        by = theano.shared(value=self.by, name='by')

        values = theano.shared(value=self.v_Whh, name='v_Whh')
        location = theano.shared(value=self.location, name='location')
        output_model = theano.shared(value=np.zeros((self.n_hidden, self.n_hidden)), name='outputmodel')
        
        def set_value_at_position(a_location, a_value, output_model):
            zeros_subtensor = output_model[a_location[0], a_location[1]]
            return T.set_subtensor(zeros_subtensor, a_value)

        outputs_info = T.zeros_like(output_model)
        result, updates = theano.scan(
            fn=set_value_at_position,
            outputs_info=outputs_info,
            sequences=[location, values])

        Whh = result[-1]

        params = [W_in, W_out, values, h0, bh, by]
        weights = [W_in, W_out, Whh, h0, bh, by]

        (params, weights) 


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

        self.params, weights = self.weight_handler.to_theano_params()
        self.W_in, self.W_out, self.W, self.h0, self.bh, self.by = weights
        self.init_structure()
        

class GraphDraw(BaseEstimator):
    def __init__(self, n_in=5, n_out=5, activation='tanh',output_type='real'):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.output_type = output_type

    def set_weights(self, Whh, Wih, Who, bh, bi):
        self.Whh = Whh
        self.Wih = Wih
        self.Who = Who
        self.bh = bh
        self.bi = bi

    def mutation(self):
        """ this is where the magic happends"""
        pass

    def get_weights(self):
        return (self.Whh, self.Wih, self.Who, self.bh, self.bi)


    def draw(self, file='neural_net.png'):
        """ represent the graph of neurons using graphviz """
        graph = pgv.AGraph(directed=False, strict=True, overlap=False)
        #draw input to hidden
        for inc_in, input, in enumerate(self.Wih):
            for inc_hidden,  weight in enumerate(input):
                if weight != 0:
                    graph.add_edge("in{}".format(inc_in), "h{}".format(inc_hidden), color="blue")  # label=t,) arrowhead='diamond')
                    node = graph.get_node("in{}".format(inc_in))
                    node.attr['style'] = 'filled'
                    node.attr['fillcolor'] = 'blue'
                    node.attr['shape'] = 'circle'
                    node.attr['label'] = ''

        #draw hidden to hidden
        for inc_hidden1, hidden_row in enumerate(self.Whh):
            for inc_hidden2, weight in enumerate(hidden_row):
                if weight != 0:
                    graph.add_edge("h{}".format(inc_hidden1), "h{}".format(inc_hidden2), color="red")  # label=t,) arrowhead='diamond')
                    node = graph.get_node("h{}".format(inc_hidden1))
                    node.attr['style'] = 'filled'
                    node.attr['fillcolor'] = 'red'
                    node.attr['shape'] = 'circle'
                    node.attr['label'] = ''

        #draw hidden to output
        for inc_hidden, hidden in enumerate(self.Who):
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

    Wih = np.ones((n_in, n_hidden))
    Whh = np.random.randint(2, size=(n_hidden, n_hidden))
    #Whh = np.identity(n_hidden)
    Who = np.ones((n_hidden, n_out))
    bh = None
    bi = None

    nn = SGRNN()
    nn.set_weights(Whh, Wih, Who, bh, bi)
    nn.draw()


if __name__ == "__main__":
    main()

import numpy as np
import random
import theano
import pygraphviz as pgv
from sklearn.base import BaseEstimator
import networkx as nx


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



        






class SGRNN(BaseEstimator):
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

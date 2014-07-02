import numpy as np
import pygraphviz as pgv


class SGRNN:
    def __init__(self):
        pass

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

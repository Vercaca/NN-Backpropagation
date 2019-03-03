import random
# import numpy as np

from activation import ActivationFunction


class NeuralNetwork:
    # def __init__(self, n_inputs=2, n_outputs, n_hiddens = 0, activ_func = 'Sigmoid'):
    #     self.n_inputs = n_inputs
    #     self.n_hiddens = n_hiddens
    #     self.n_outputs = n_outputs
    #     self.activ_func = activ_func
    #     self.layers = []
    def __init__(self, activ_func = 'Sigmoid', learning_rate='0.01'):
        self.activ_func = ActivationFunction(types=activ_func)
        self.layers = []
        self.learning_rate = learning_rate

    def add_layer(self, n_inputs, n_neurons):
        layer = NeuralLayer(n_inputs, n_neurons, self.activ_func)
        self.layers.append(layer)
        print(layer)
        # self.n_hiddens += 1
        return None

    def feed_forward(self, inputs):
        for i, layer in enumerate(self.layers):
            inputs = layer.feed_forward(inputs) # next_input = previous_output
            #debug
            print('Layer {}, Output: {}'.format(i+1, inputs))

        return None

    def feed_backward(self, targets): # backpropagating
        # calculate errors
        
        return

    def update_weights(self):

        return

    def calcuate_error(self, predictions, targets):

    def train(self, dataset, n_iterations=2):
        for i in range(n_iterations):
            print('\n>> Iteration #{}'.format(i+1))
            for j, (inputs, outputs) in enumerate(dataset):
                print('data # {}'.format(j))
                self.feed_forward(inputs)
                self.feed_backward(outputs)
                self.update_weights()
        return

    def test(self):
        return

class NeuralLayer:
    __counter = 0

    def __init__(self, n_inputs, n_neurons, activ_func):
        # NeuralLayer.__counter += 1
        self.__counter = NeuralLayer.__counter = NeuralLayer.__counter + 1
        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs, activ_func))

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def __repr__(self):
        return ('Layer {}  # of neurons: {}'.format(self.__counter, len(self.neurons)))


class Neuron:
    def __init__(self, n_weights, activ_func='Sigmoid', bias=1):
        self.__weights = [random.random() for i in range(n_weights)] # np.random.rand(n_weights)#
        self.__bias = bias
        self.__output = 0
        self.__inputs = []
        self.activation_function = activ_func

    def calculate_output(self, inputs):
        n_weights = len(self.__weights)
        if len(inputs) != n_weights:
            raise Exception('wrong inputs number')

        output = sum([inputs[i] * self.__weights[i] for i in range(n_weights)])

        a_output = self.activation_function.func(output+self.__bias)

        # fill the variables
        self.__inputs = inputs
        self.__output = a_output

        return a_output


def demo():
    # [(inputs, outputs)]
    dataset = [
    [(1, 0), 1],
    [(0, 0), 0]
    ]

    nn = NeuralNetwork(learning_rate='0.01')
    nn.add_layer(n_inputs=2, n_neurons=3)
    nn.add_layer(n_inputs=3, n_neurons=1)
    nn.train(dataset=dataset)


if __name__ == '__main__':
    demo()

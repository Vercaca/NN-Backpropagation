from neural_network import *

# example of XOR
train_dataset = [
            [(1, 0), [1]],
            [(0, 0), [0]],
            [(0, 1), [1]],
            [(1, 1), [0]]
            ]

nn = NeuralNetwork(learning_rate=0.1, debug=False)
nn.add_layer(n_inputs=2, n_neurons=3)
nn.add_layer(n_inputs=3, n_neurons=1)

nn.train(dataset=train_dataset, n_iterations=100, print_error_report=True)

# test
test_dataset = [
    [(1, 0), [1]],
    [(0, 0), [0]]
]
nn.test(test_dataset)

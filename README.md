# Neural Network with BackPropagation
Implement a simple Neural network trained with backprogation in Python3.

## How to train a supervised Neural Network?
1. Feed Forward
2. Feed Backward * (BackPropagation)
3. Update Weights
Iterating the above three steps

![FeedForward vs. FeedBackward](https://cdn-images-1.medium.com/max/1600/1*q1M7LGiDTirwU-4LcFq7_Q.png)
Figure 1. FeedForward vs. FeedBackward (by Mayank Agarwal)


## Description of BackPropagation (小筆記)
Backpropagation is the implementation of gradient descent in multi-layer neural networks. Since the same training rule recursively exists in each layer of the neural network, we can calculate the contribution of each weight to the total error inversely from the output layer to the input layer, which is so-called backpropagation.


### Gradient Descent (Optimization)
Gradient descent is a first-order iterative optimization algorithm, which is used to find the local minima or global minima of a function. The algorithm itself is not hard to understand, which is:

1. Starting from a point on the graph of a function;
2. Find a direction ▽F(a) from that point, in which the function decreases fastest;
3. Go (down) along this direction a small step γ, got to a new point of a+1;
By iterating the above three steps, we can find the local minima or global minima of this function.

### Stochastic Gradient Descent (SGD)
The advantage of this method is that the gradient is accurate and the function converges fast. But when the training dataset is enormous, the evaluation of the gradient from all data points becomes expensive and the training time can be very long.

Another method is called stochastic gradient descent, which samples (with replacement) a subset (one or more) of training data to calculate the gradient. 


## My Implementation
### How to Learn?
Stochastic Gradient Descent

每次迭代:

Input為一筆資料的features，模型透過feed forward運算出predict_Y (outputs)

將所有Outputs與其相對之targets比較，利用Gradient Descent找出每個Neuron中的weights的改變方向

修正Weights，下一筆資料用新的權重來進行prediction，如此類推直至收斂。

## Version
1.0

## Requirements
```
Python==3
```
## Coding Description
#### Activation Function: Sigmoid
#### Error Minimization: Gradient Descent
#### Error Function: Mean Square Error



## Demo 
(demo_1_xor.py)
Here we used XOR dataset for our first demo.

#### Input datasets
```
# dataset = [(inputs), (outputs)]
train_dataset = [
    [(1, 0), [1]],
    [(0, 0), [0]],
    [(0, 1), [1]],
    [(1, 1), [0]]
]

test_dataset = [
    [(1, 0), [1]],
    [(0, 0), [0]]
]
```

#### Build the NN model with 1 hidden layer (2-3-1) 
(n_input(2) -> n_hiddens(3,) --> n_output(1))
```
nn = NeuralNetwork(learning_rate=0.1, debug=False)
nn.add_layer(n_inputs=2, n_neurons=3)
nn.add_layer(n_inputs=3, n_neurons=1)
```

#### Train
```
nn.train(dataset=train_dataset, n_iterations=100, print_error_report=True)
```

#### Test
```
nn.test(test_dataset)
```

## Remark
It is just a rough neural netowrk with bp.

### Future Logs
- Hidden Layers can be added at initial.
- More Activation Functions


## References
Concepts:

- [類神經網路--BackPropagation 詳細推導過程 By Mark Chang](http://cpmarkchang.logdown.com/posts/277349-neural-network-backward-propagation)
- [Gradient Descent and Backpropagation By Ken Chen](https://www.linkedin.com/pulse/gradient-descent-backpropagation-ken-chen/)
- [gradient descent optimization algorithms By Tommy Huang](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7)
- [Delta Rule By Wikipedia](https://en.wikipedia.org/wiki/Delta_rule)
- [Figure_1 Feedforward vs. Feedbackward](https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199)

Coding:

- [Coding_Reference(GitHub) #1](https://github.com/del680202/MachineLearning-memo)
- [Coding_Reference(GitHub) #2](https://github.com/jgabriellima/backpropagation)



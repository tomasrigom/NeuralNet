import numpy as np
import random

class Network():

    def __init__(self, neurons_layers, activationf, costf):

        '''
        Initialize network with:

            - self.num_layers (int) : number of layers in the network (including the input and output layers)
            - neurons_layers / self.neurons_layers (list) : number of neurons in each layer
            - activationf / self.activationf (function) : activation function of the neurons
            - costf / self.costf (function) : cost function used to compute the cost

        '''
        self.num_layers = len(neurons_layers)
        self.neurons_layers = neurons_layers
        self.initialize_weights()
        self.activationf = activationf
        self.costf = costf

    def initialize_weights(self):
        '''
        Initialize the weights and biases of the neural network in two different ways:

            - The weights are initialized using a normal distribution with mean zero and standard deviation 1 divided by the square root of the number of neurons connected to that one
            - The biases are initialised simply using a normal distribution of mean zero and standard deviation 1

        '''
        self.biases = [np.random.randn(neurons,1) for neurons in self.neurons_layers[1:]]
        self.weights = [np.random.randn(neurons, neurons_prev)/np.sqrt(neurons_prev) for neurons, neurons_prev in zip(self.neurons_layers[1:], self.neurons_layers[:-1])]

    def feedforward(self, a):
        '''
        Return the output of the network based on the 

            - a (list) : Activation values of the neurons in the last layer

        For each connection do the perceptron calculation w * a + b, and pass it through the activation function
        '''
        for w, b in zip(self.weights, self.biases):
            a = self.activationf(np.dot(w,a) + b)
        return a
    
    def updateparams(self, eta, ):
        
    
    def backprop(self, x, y):
        '''
        Obtains the value for the deltas using backpropagation algorithm for a single training example

            - x (numpy array) : raw data
            - y (int) : expected result
        '''
        # Initialize the gradients for the weights and biases as zero
        nablab = [np.zeros(bias.shape) for bias in self.biases]
        nablaw = [np.zeros(weight.shape) for weight in self.weights]

        # Initialize the lists that will store the activations of all the layers and the intermediate perceptron sums z = a*w + b
        a_layers = [x]
        z_layers = []

        # We use the introduced data to feedforward and get a result, storing the zs and activations
        for n, (w, b) in enumerate(zip(self.weights, self.biases)):
            z_layers.append(np.dot(w,a_layers[n]) + b)
            a_layers.append(self.activationf(z_layers[n]))

        # Now we get the delta of the final layer and propagate backwards
        delta = (self.costf).delta(z_layers[-1])
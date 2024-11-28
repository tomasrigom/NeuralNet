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
            a = (self.activationf).evaluate(np.dot(w,a) + b)
        return a
    
    def stochastic_gradient_descent(self, trainingdata, batch_size, epochs, eta):
        '''
        Train the model using stochastic gradient descent

            - trainingdata (list of tuples (x, y)) : contains the (data, label) tuples to train the model
            - batch_size (int) : number of training examples per minibatch used
            - epochs (int) : number of epochs during which the model will be trained
            - eta (float) : learning rate
        '''

        n = len(trainingdata)
        for epoch in range(epochs):
            # Shuffle the data, then divide it into subsets of size batch_size
            random.shuffle(trainingdata)
            mini_batches = [trainingdata[i:i+batch_size] for i in range(0,n,batch_size)]

            for minibatch in mini_batches:
                self.updateparams_minibatch(minibatch,eta)
    
    def updateparams_minibatch(self, minibatch, eta):
        '''
        Updates the parameters using the input mini-batch of data, by getting the derivatives of the cost function using the function 'backprop'

            - batch (list of tuples (x,y)) : contains the (data, label) tuples for the current mini-batch, which we will use to update the parameters
            - eta (float) : learning rate of the algorithm
        '''
        #Initialize the sum of the partial derivatives of the cost with respect to the weights and the biases as zero
        sum_pcost_pw = [np.zeros(w.shape) for w in self.weights]
        sum_pcost_pb = [np.zeros(b.shape) for b in self.biases]

        # Add up the aforementioned derivatives for every training example in the mini-batch
        for batch in minibatch:
            pcost_pw, pcost_pb = self.backprop(batch[0],batch[1])
            sum_pcost_pw = [sum_prev + new_term for sum_prev, new_term in zip(sum_pcost_pw, pcost_pw)]
            sum_pcost_pb = [sum_prev + new_term for sum_prev, new_term in zip(sum_pcost_pb, pcost_pb)]

        # Modify the parameters according to the obtained data
        self.weights = [w - eta * sum_learned_w / len(minibatch) for w, sum_learned_w in zip(self.weights, sum_pcost_pw)]
        self.biases = [b - eta * sum_learned_b / len(minibatch) for b, sum_learned_b in zip(self.biases, sum_pcost_pb)]

    
    def backprop(self, x, y):
        '''
        Obtains the value for the deltas using backpropagation algorithm for a single training example

            - x (numpy array) : raw data
            - y (int) : expected result
        '''
        # Initialize the partial derivatives of the cost function with respect to the weights and biases to zero
        pcost_pw = [np.zeros(weight.shape) for weight in self.weights]
        pcost_pb = [np.zeros(bias.shape) for bias in self.biases]

        # Initialize the lists that will store the activations of all the layers and the intermediate perceptron sums z = a*w + b
        a_layers = [x]
        z_layers = []

        # We use the introduced data to feedforward and get a result, storing the zs and activations
        for n, (w, b) in enumerate(zip(self.weights, self.biases)):
            z_layers.append(np.dot(w,a_layers[n]) + b)
            a_layers.append((self.activationf).evaluate(z_layers[n]))

        # Now we get the delta of the final layer and propagate backwards
        delta = (self.costf).delta(z_layers[-1], a_layers[-1], y)
        pcost_pb[-1] = delta
        pcost_pw[-1] = np.dot(delta,a_layers[-2].transpose())

        # Loop from the second to last layer, backwards
        for l in range(2,self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * (self.activationf).derivative(z_layers[-l])
            pcost_pb[-l] = delta
            pcost_pw[-l] = np.dot(delta, a_layers[-l-1].transpose())

        # Lastly return the resulting matrices
        return pcost_pw, pcost_pb
    
    def accuracy(self, data):
        '''
        Returns the accuracy of the model in predicting the labels of the current dataset

            - data (list of tuples) : tuples (data, label) on which the model will be evaluated
        
        where data is the raw data (image), and label is a list representing which neuron represents the correct result (so all zeros except for one)
        '''
        return np.sum([np.argmax(self.feedforward(rawdata)) == np.argmax(label) for rawdata, label in data])/len(data)


import numpy as np
import random
import json
import sys

import time

################################################################################################################
# Activation functions
################################################################################################################

class LinearActivation():
    '''
    Class for the linear activation function, including the function evaluation and its derivative
    '''
    @staticmethod
    def evaluate(z):
        return z
    
    @staticmethod
    def derivative(z):
        return np.ones_like(1)

class SigmoidActivation():
    '''
    Class for the sigmoid activation function, including the function evaluation, and its derivative
    '''
    @staticmethod
    def evaluate(z):
        return 1 / (1+np.exp(-z))
    
    @staticmethod
    def derivative(z):
        sig = SigmoidActivation.evaluate(z)
        return sig*(1-sig)
    
class TanhActivation():
    '''
    Class for the hyperbolic tangent activation function, including the function evaluation, and its derivative

        IMPORTANT NOTE : This is simply a rough implementation of the tanh neurons, potential normalization of the inputs and/or outputs might need to be done to use this neuron activation

    '''
    @staticmethod
    def evaluate(z):
        return np.tanh(z)
    
    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z)**2
    
class ReluActivation():
    '''
    Class for the ReLU activation function, including the function evaluation, and its derivative

        IMPORTANT NOTE : This is simply a rough implementation of the relu neurons, potential normalization of the inputs and/or outputs might need to be done to use this neuron activation

    '''    
    @staticmethod
    def evaluate(z):
        return np.maximum(0,z)
    
    @staticmethod
    def derivative(z):
        return np.where(z > 0, 1, 0)

    
class SoftmaxActivation():
    '''
    Class for the softmax output, including the function evaluation, and its derivative
    '''
    @staticmethod
    def evaluate(z):
        # The maximum value is subtracted for numerical stability: this will not alter the result
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        # Note the sum is over axis=1, since the first index is kept for separating different examples or batches
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def derivative(z,jacobian=False):
        # NOTE: The distinction between returning the jacobian or not jacobian is only relevant for suboptimal output-cost choices. Nevertheless, i included it for completion
        f = SoftmaxActivation.evaluate(z)
        if jacobian:
            jacobians = [np.zeros((example.shape[1],example.shape[1])) for example in z]
            for softmax_vec, softmax_J in zip(f, jacobians):
                for i in range(softmax_vec.shape[0]):
                    for j in range(softmax_vec.shape[0]):
                        if i==j:
                            softmax_J[i,j] = softmax_vec[i]*(1-softmax_vec[i])
                        else:
                            softmax_J[i,j] = -softmax_vec[i]*softmax_vec[j]
            return softmax_J
        
        else:
            return f*(1-f)

################################################################################################################
# Cost functions
################################################################################################################

class LinearCost():
    '''
    Class for the linear cost function, including the function evaluation, and the delta value from the output layer
    '''
    @staticmethod
    def evaluate(a, y):
        # Note the sum is over axis=1, since the first index is kept for separating different examples or batches
        return np.sum(np.abs(a-y), axis = 1)
    
    @staticmethod
    def delta(z,a,y,activationf):
        if activationf.__class__.__name__ == 'SoftmaxActivation':
            return np.array([np.sum(J,axis=1) for J in activationf.derivative(z, jacobian=True)]).reshape(a.shape)
                
        else:
            return np.ones_like(a) * activationf.derivative(z)


class QuadraticCost():
    '''
    Class for the quadratic cost function, including the function evaluation, and the delta value from the output layer
    '''
    @staticmethod
    def evaluate(a, y):
        return 0.5 * np.sum((a-y)**2, axis = 1)
    
    @staticmethod
    def delta(z,a,y,activationf):
        if activationf.__class__.__name__ == 'SoftmaxActivation':
            return np.array([np.dot(J, aidiff) for J, aidiff in zip(activationf.derivative(z, jacobian=True), a-y)]).reshape(a.shape)
        
        else:
            return (a - y) * activationf.derivative(z)

        
class CrossEntropyCost():
    '''
    Class for the cross-entropy cost function, including the function evaluation, and the delta value from the output layer
    '''

    def __init__(self):
        self.flagwarned = 0

    @staticmethod
    def evaluate(a,y):
        epsilon = 1e-14  # Small constant to avoid log(0)
        a = np.clip(a, epsilon, 1 - epsilon)  # Clip a to be in [epsilon, 1 - epsilon]

        return np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a), axis=1)
    
    def delta(self,z,a,y,activationf):
        # If the activation function of the output layer is chosen a sigmoid, the learning slowdown problem is solved: this is optimal
        if activationf.__class__.__name__ == 'SigmoidActivation':
            return a - y
        
        elif activationf.__class__.__name__ == 'SoftmaxActivation':
            return np.array([np.dot(J, aidiff) for J, aidiff in zip(activationf.derivative(z, jacobian=True), (a - y)/ a*(1 - a))]).reshape(a.shape)
        
        else:
            if self.flagwarned == 0:
                print("######################################################\nWARNING: CHOICE OF OUTPUT LAYER ACTIVATION AND COST FUNCTION INEFFICIENT\n######################################################")
                self.flagwarned = 1
            return (a - y) * activationf.derivative(z) / a*(1 - a)


class LoglikelyCost():
    '''
    Class for the log-likelyhood cost function, including the function evaluation, and the delta value from the output layer

        IMPORTANT NOTE: For the usage of LogLikelyCost.delta() in combination with the softmax activation function, it MUST be the case that the 'y' vector is all zeros except for the correct element (label), which must be 1

    '''
    @staticmethod
    def evaluate(a,y):
        return -np.log(a[np.arange(a.shape[0]), np.argmax(y, axis=1)])
    
    def delta(self,z,a,y,activationf):
        # If the activation function of the output layer is chosen a sigmoid, the learning slowdown problem is solved: this is optimal
        if activationf.__class__.__name__ == 'SoftmaxActivation':
            return a - y
        
        else:
            result = 0
            if self.flagwarned == 0:
                print("######################################################\nWARNING: CHOICE OF OUTPUT LAYER ACTIVATION AND COST FUNCTION INEFFICIENT\n######################################################")
                self.flagwarned = 1
            result[np.arange(a.shape[0]), np.argmax(y, axis=1)] = -activationf.derivative(z) / a[np.arange(a.shape[0]), np.argmax(y, axis=1)]

        return result


################################################################################################################
# Mappings of the activation and cost function classes in a dictionary, useful for later loading of a model (see Network.load(filename) below)
################################################################################################################

activationf_mapping = {'LinearActivation':LinearActivation, 'SigmoidActivation':SigmoidActivation, 'TanhActivation':TanhActivation, 'ReluActivation':ReluActivation, 'SoftmaxActivation':SoftmaxActivation}
costf_mapping = {'LinearCost':LinearCost, 'QuadraticCost':QuadraticCost, 'CrossEntropyCost':CrossEntropyCost, 'LoglikelyCost':LoglikelyCost}

################################################################################################################
#    NETWORK CLASS DEFINITION
################################################################################################################

class Network():

    def __init__(self, neurons_layers, 
                 activationf_hidden = SigmoidActivation(), 
                 activationf_output = SigmoidActivation(), 
                 costf = CrossEntropyCost(),
                 reg = 'L2',
                 momentum = False,
                 print_terminal = True):

        '''
        Initialize network with inputs:

            - neurons_layers (list) : number of neurons in each layer
            - activationf_hidden (class) : activation function class (defined above) of the neurons in the hidden layers
            - activationf_output (class) : activation function class (defined above) of the neurons in the output layer
            - costf (class) : cost function class (defined above) used to compute the cost
            - reg (string or None) : regularization type to use (None, L1, L2)
            - momentum (bool) : Indicates whether to use momentum-based gradient descent
            - print_terminal (bool): Indicates whether to print the logging on to the terminal

        and attributes:

            - self.neurons_layers (list) : same value and function as neurons_layers
            - self.num_layers (int) : number of layers in the network (including the input and output layers)
            - self.initialize_weights initialises the weights and biases of the model as described in "self.initialize_weights()"
                * self.weights (list of np.arrays) : weights of each layer connections, where the position (i,j,k) represents the weight of the connection between neuron j from layer i to the neuron k from layer i-1
                * self.biases (list of np.arrays) : biases of each neuron in each layer, where the position (i,j) represents the bias of neuron j in layer i
            - self.activationf (list of classes) : activation function of the neurons in each layer
            - self.costf (class) : same value and function as cost
            - self.reg (string or None) : same value and function as reg
            - self.momentum (bool) : same value and function as bool. Initializing this parameter as True creates two new attributes for this model:
                * self.vw : 'Velocity' of the weights
                * self.vb : 'Velocity' of the biases
            - self.print_terminal (bool): same as print_terminal input
        '''
        self.neurons_layers = neurons_layers
        self.num_layers = len(neurons_layers)

        if self.num_layers > 2:
            self.activationf = [activationf_hidden] * (self.num_layers-2) + [activationf_output]
        elif self.num_layers == 2:
            self.activationf = [activationf_output]
        else:
            raise Exception("Number of layers must be larger than 2")
        
        self.initialize_weights()
        
        self.costf = costf
        self.reg = reg
        self.momentum = momentum
        if self.momentum:
            self.vw = [np.zeros_like(w) for w in self.weights]
            self.vb = [np.zeros_like(b) for b in self.biases]

        self.print_terminal = print_terminal

        if self.print_terminal:
            print(f"\n#### INITIALIZING NEURAL NETWORK WITH SETTINGS: ####\n   Number of layers: {self.num_layers}\n   Neurons per layer: {self.neurons_layers}\n   Activation function in hidden layers: {activationf_hidden.__class__.__name__}\n   Activation function in output layer: {activationf_output.__class__.__name__}\n   Cost function: {self.costf.__class__.__name__}")
            print(f"   No regularization") if not self.reg else print(f"   Regularization type: {self.reg}")
            print(f"   Ordinary gradient descent") if not self.momentum else print(f"   Momentum-based gradient descent\n")


    def initialize_weights(self):
        '''
        Initialize the weights and biases of the neural network.
            - Layers with linear, sigmoid, or tanh activations will be initialised using Xavier weight initialization. 
            - Layers with ReLU activation will be initialised using He initialisation.
            - If the output layer is set to softmax, the corresponding weights and biases will be initialized to zero
        '''
        self.biases = []
        self.weights = []
        for layer,activation in enumerate(self.activationf):
            if activation.__class__.__name__ == 'LinearActivation' or activation.__class__.__name__ == 'SigmoidActivation' or activation.__class__.__name__ == 'TanhActivation':
                self.biases.append(np.random.randn(self.neurons_layers[layer+1],1))
                self.weights.append(np.random.randn(self.neurons_layers[layer+1], self.neurons_layers[layer])/np.sqrt(self.neurons_layers[layer]))

            elif activation.__class__.__name__ == 'ReluActivation':
                self.biases.append(np.random.randn(self.neurons_layers[layer+1],1))
                self.weights.append(np.random.randn(self.neurons_layers[layer+1], self.neurons_layers[layer]) * np.sqrt(2 / self.neurons_layers[layer]))

            elif activation.__class__.__name__ == 'SoftmaxActivation':
                self.biases.append(np.zeros((self.neurons_layers[layer+1],1)))
                self.weights.append(np.zeros((self.neurons_layers[layer+1], self.neurons_layers[layer])))

            else:
                raise Exception("No implemented initialization for the activation {activation.__class__.__name__}")


    def feedforward(self, a):
        '''
        Return the output of the network from the input activations 

            - a (array of 1D arrays) : Activation values of the neurons in the input layer, for several training examples

        For each connection do the perceptron calculation w * a + b, and pass it through the activation function to the output. This is done in a vectorized way across all examples in a.
        '''
        for w, b, activation in zip(self.weights, self.biases, self.activationf):
            a = activation.evaluate(np.dot(a, w.T) + b.T)
        return a
    

    def stochastic_gradient_descent(self, trainingdata, evaluationdata = None,
                                    batch_size = 100, epochs = 400, eta = 0.1, eta_var = None, lmbda = 0.1, mu = 0.1,
                                    dynamic_stop = (15,1e-2),
                                    monitor_training_cost = False,
                                    monitor_training_accuracy = False,
                                    monitor_evaluation_cost = False,
                                    monitor_evaluation_accuracy = False):
        '''
        Train the model using stochastic gradient descent

            - trainingdata (list of tuples) : contains the (data, label) tuples to train the model
                data is a numpy array
                labels is a numpy array
            - evaluationdata (list of tuples) : contains the (data, label) tuples to train the model
                NOTE: This is not the same as the testing data. It is an intermediate set to test hyperparameters (batch_size, eta, lambda, mu, ...)
            - batch_size (int) : number of training examples per minibatch used
            - epochs (int) : number of epochs during which the model will be trained
            - eta (float or function) : learning rate. It can be either a constant or a function of the variable specified in the parameters eta_var
            - eta_var (str): specifies if the learning rate depends on the accuracy on the evaluation data ("evacc") or on the epoch ("epoch")
            - lmbda (float) : regularization parameter
            - mu (float) : momentum co-efficient (only relevant if self.momentum = True)
            - dynamic_stop (tuple (int, float)) : If the first element is set to a positive value, it will stop training when dynamic_stop[0] epochs pass without the cost decreasing by more than dynamic_stop[1]
            - monitor_training_cost (bool) : If True, logs the cost on the training data at each epoch (and prints if self.print_terminal is True)
            - monitor_training_accuracy (bool) : If True, logs the accuracy on the training data at each epoch (and prints if self.print_terminal is True)
            - monitor_evaluation_cost (bool) : If True, logs the cost on the evaluation data at each epoch (and prints if self.print_terminal is True)
            - monitor_evaluation_accuracy (bool) : If True, logs the accuracy on the evaluation data at each epoch (and prints if self.print_terminal is True)
        '''
        t_begin = time.time()
        n = len(trainingdata)
        training_cost, training_accuracy, evaluation_cost, evaluation_accuracy= [], [], [], []

        for epoch in range(epochs):
            t_epoch = time.time()
            # At each epoch, shuffle the data, then divide it into subsets of size batch_size
            random.shuffle(trainingdata)
            mini_batches = [trainingdata[i:i+batch_size] for i in range(0, n, batch_size)]

            # For each batch update the parameters using backpropagation
            if callable(eta):
                if evaluationdata==None:
                    raise Exception("If eta is made variable, an evaluation set is needed to evaluate it")
                
                if eta_var == 'evacc':
                    eta_value = eta(1. - self.accuracy(evaluationdata)/100.)
                elif eta_var == 'epoch':
                    eta_value = eta(epoch)
                else:
                    raise Exception("Variable eta not compatible with its dependance")
                
                if self.print_terminal:
                    print(f"Value of eta: {eta_value:.3f}")

                for minibatch in mini_batches:
                    self.updateparams_minibatch(minibatch, eta_value, n, lmbda, mu)

                if self.print_terminal:
                    print(f"Epoch {epoch} complete in {time.time()-t_epoch:.1f} seconds")

            elif not callable(eta):
                for minibatch in mini_batches:
                    self.updateparams_minibatch(minibatch, eta, n, lmbda, mu)

                if self.print_terminal:
                    print(f"Epoch {epoch} complete in {time.time()-t_epoch:.1f} seconds")

            # Monitor the cost function on the training data
            if monitor_training_cost:
                epoch_cost = self.cost(trainingdata,lmbda)
                training_cost.append(epoch_cost)

                if self.print_terminal:
                    print(f"    Training cost: {epoch_cost:.2f}")

            # Monitor the accuracy on the training data
            if monitor_training_accuracy:
                epoch_accuracy = self.accuracy(trainingdata)
                training_accuracy.append(epoch_accuracy)

                if self.print_terminal:
                    print(f"    Training accuracy: {epoch_accuracy:.2f}%")

            # Monitor the cost function on the evaluation data
            if monitor_evaluation_cost and evaluationdata:
                epoch_cost = self.cost(evaluationdata,lmbda)
                evaluation_cost.append(epoch_cost)

                if self.print_terminal:
                    print(f"    Evaluation cost: {epoch_cost:.2f}")
            
            # Monitor the accuracy on the evaluation data
            if monitor_evaluation_accuracy and evaluationdata:
                epoch_accuracy = self.accuracy(evaluationdata)
                evaluation_accuracy.append(epoch_accuracy)

                if self.print_terminal:
                    print(f"    Evaluation accuracy: {epoch_accuracy:.2f}%")

            # Check if cost has stopped decreasing
            if dynamic_stop[0] >= 0:
                if not monitor_evaluation_accuracy and evaluationdata:
                    epoch_accuracy = self.accuracy(evaluationdata)
                    evaluation_accuracy.append(epoch_accuracy)

                if epoch >= dynamic_stop[0] and evaluation_accuracy[epoch] - evaluation_accuracy[epoch - dynamic_stop[0]] < dynamic_stop[1]:
                    if self.print_terminal:
                        print(f"Difference in {dynamic_stop[0]} epochs smaller than {dynamic_stop[1]} threshold: {evaluation_accuracy[epoch] - evaluation_accuracy[epoch - dynamic_stop[0]]:.3f}")

                    break
        
        if self.print_terminal:
            print(f"\nModel trained in: {time.time() - t_begin:.1f} seconds\n")

        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy
    

    def updateparams_minibatch(self, minibatch, eta, n, lmbda, mu):
        '''
        Updates the parameters using the input mini-batch of data, by getting the derivatives of the cost function using the function 'backprop'

            - minibatch (list of tuples (x,y)) : contains the (data, label) tuples for the current mini-batch, which we will use to update the parameters
            - eta (float) : learning rate of the algorithm
            - n (int) : size training data (from which the minibatch is randomly chosen)
            - lmbda (float) : regularization parameter
            - mu (float) : friction parameter (only relevant if self.momentum = True)
        '''
        rawdata, labels = zip(*minibatch)
        rawdata = np.array(rawdata)
        labels = np.array(labels)

        # Some calculations to avoid re-calculating quantities
        minibatch_size = len(minibatch)
        eta_scaled = eta/minibatch_size

        #Initialize the sum of the partial derivatives of the cost with respect to the weights and the biases as zero
        sum_pcost_pw = [np.zeros_like(w) for w in self.weights]
        sum_pcost_pb = [np.zeros_like(b) for b in self.biases]

        # Add up the aforementioned derivatives for every training example in the mini-batch
        pcost_pw, pcost_pb = self.backprop(rawdata,labels)
        sum_pcost_pw = [np.sum(pcpw, axis=0) for pcpw in pcost_pw]
        sum_pcost_pb = [np.sum(pcpb, axis=0) for pcpb in pcost_pb]
       
        # Now the function is different depending on whether we use normal or momentum-based gradient descent
        if self.momentum:
            # L1 regularization
            if self.reg == 'L1':
                # Update velocities of the weights
                self.vw = [mu*v - eta_scaled*(np.sign(w)*lmbda/n + sum_learned_w) for v, w, sum_learned_w in zip(self.vw, self.weights, sum_pcost_pw)]

            # L2 regularization
            elif self.reg == 'L2':
                # Update velocities of the weights
                self.vw = [mu*v - eta_scaled*(w*lmbda/n + sum_learned_w) for v, w, sum_learned_w in zip(self.vw, self.weights, sum_pcost_pw)]

            # No regularization
            else:
                # Update velocities of the weights
                self.vw = [mu*v - eta_scaled * sum_learned_w for v, sum_learned_w in zip(self.vw, sum_pcost_pw)]

            # Update velocities of the biases
            self.vb = [mu*v - eta_scaled * sum_learned_b[:,np.newaxis] for v, sum_learned_b in zip(self.vb, sum_pcost_pb)]

            # Update weights and biases
            self.weights = [w + v for v, w in zip(self.vw, self.weights)]
            self.biases = [b + v for v, b in zip(self.vb, self.biases)]

        else:
            # L1 regularization
            if self.reg == 'L1':
                # Update the weights
                self.weights = [w - eta_scaled*(np.sign(w)*lmbda/n + sum_learned_w) for w, sum_learned_w in zip(self.weights, sum_pcost_pw)]

            # L2 regularization
            elif self.reg == 'L2':
                # Update weights
                self.weights = [(1 - eta*lmbda/n)*w - eta_scaled * sum_learned_w for w, sum_learned_w in zip(self.weights, sum_pcost_pw)]

            # No regularization
            else:
                # Update weights
                self.weights = [w - eta_scaled * sum_learned_w for w, sum_learned_w in zip(self.weights, sum_pcost_pw)]

            # Update biases
            self.biases = [b - eta_scaled * sum_learned_b for b, sum_learned_b in zip(self.biases, sum_pcost_pb)]

    
    def backprop(self, x, y):
        '''
        Obtains the value for the deltas using backpropagation algorithm for a single training example

            - x (numpy array of 1D numpy arrays) : array containing the raw data of the different examples
            - y (numpy array of 1D numpy arrays) : array containing one-hot vectors indicating the labels orresponding to each example in x
        '''
        # Initialize the partial derivatives of the cost function with respect to the weights and biases to zero
        pcost_pw = [np.zeros((x.shape[0], *w.shape), dtype = np.float32) for w in self.weights]
        pcost_pb = [np.zeros((x.shape[0], *b.shape), dtype = np.float32) for b in self.biases]

        # Initialize the lists that will store the activations of all the layers and the intermediate perceptron sums z = a*w + b
        a_layers = [x]
        z_layers = []

        # We use the introduced data to feedforward and get a result, storing the zs and activations
        for n, (w, b) in enumerate(zip(self.weights, self.biases)):
            z_layers.append(np.dot(a_layers[n], w.T) + b.T)
            a_layers.append((self.activationf[n]).evaluate(z_layers[n]))

        # Now we get the delta of the final layer
        delta = (self.costf).delta(z_layers[-1], a_layers[-1], y, self.activationf[-1])
        pcost_pb[-1] = delta
        pcost_pw[-1] = np.einsum('ij,ik->ijk', delta, a_layers[-2])

        # Propagate backwards: loop from the second to last layer, backwards
        for l in range(2,self.num_layers):
            delta = np.dot(self.weights[-l+1].T,delta.T).T * (self.activationf[-l]).derivative(z_layers[-l])
            pcost_pb[-l] = delta
            pcost_pw[-l] = np.einsum('ij,ik->ijk', delta, a_layers[-l-1])

        # Lastly return the resulting matrices
        return pcost_pw, pcost_pb
    
    def accuracy(self, data):
        '''
        Returns the accuracy (%) of the model in predicting the labels of the current dataset

            - data (list of tuples) : tuples (rawdata, label) on which the model will be evaluated
        
        where rawdata is the raw data, and label is a list representing which neuron represents the correct result (so all zeros except for one)
        '''
        rawdata, labels = zip(*data)
        rawdata = np.array(rawdata)
        labels = np.array(labels)

        return 100. * np.sum(np.argmax(self.feedforward(rawdata), axis=1) == np.argmax(labels, axis=1))/rawdata.shape[0]

    def cost(self, data, lmbda):
        '''
        Returns the total cost of running the model on the data
        The arguments of the method are analogous to those from self.accuracy
        '''
        # Get the output activations of each training example
        rawdata, labels = zip(*data)
        rawdata = np.array(rawdata)
        labels = np.array(labels)

        activations = self.feedforward(rawdata)

        # Calculate the cost
        cost = np.sum(self.costf.evaluate(activations, labels))
        
        if self.reg == 'L1':
            cost += 0.5 * lmbda * sum(np.sum(abs(w)) for w in self.weights) 
        elif self.reg == 'L2':
            cost += 0.5 * lmbda * sum(np.sum(w**2) for w in self.weights) 

        return cost/len(data)
    
    def savemodel(self, filename):
        '''
        Saves the model to a JSON file. The input parameter is:
        
            - filename (string): the name of the output file which will contain the model. Therefore, it will be saved as 'filename'

        Note the extension '.json' has to be included in the filename
        '''
        # Data is saved as a dictionary, as per usual with JSON files
        outmodel = {'neurons_layers':self.neurons_layers, 
                    'weights':[w.tolist() for w in self.weights], 
                    'biases':[b.tolist() for b in self.biases], 
                    'activationf':[func.__class__.__name__ for func in self.activationf],
                    'costf':self.costf.__class__.__name__ }
        
        # Save to file
        with open(filename,'w') as outfile:
            json.dump(outmodel,outfile)
        
        if self.print_terminal:
            print(f"Model successfully saved at {filename}")

    def loadmodel(self, filename):
        '''
        Contrary process to the one from self.loadmodel: loading the model in 'filename' into the current object
        The input parameters are analogous to those from self.savemodel
        '''

        with open(filename,'r') as infile:
            # Load dictionary with the data
            inmodel = json.load(infile)

        # Update the model's attributes to those from the file
        self.neurons_layers = inmodel['neurons_layers']
        self.num_layers = len(inmodel['neurons_layers'])
        self.weights = [np.array(w) for w in inmodel['weights']]
        self.biases = [np.array(b) for b in inmodel['biases']]
        self.activationf = [activationf_mapping[activationf] for activationf in inmodel['activationf']]
        self.costf = costf_mapping[inmodel['costf']]

        if self.print_terminal:
            print(f"Model {filename} successfully loaded")

import numpy as np
import random
import json
import sys

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
        return 1

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
        return np.exp(z) / sum(np.exp(z))
    
    @staticmethod
    def derivative(z):
        ############################################3CORRCT THIS###############################################
        f = SoftmaxActivation.evaluate(z)
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
        return np.sum(a-y)
    
    @staticmethod
    def delta(z,a,y,activationf):
        return np.ones(len(a)) * activationf.derivative(z)

class QuadraticCost():
    '''
    Class for the quadratic cost function, including the function evaluation, and the delta value from the output layer
    '''
    @staticmethod
    def evaluate(a, y):
        return 0.5 * np.sum((a-y)**2)
    
    @staticmethod
    def delta(z,a,y,activationf):
        return (a - y) * activationf.derivative(z)

        
class CrossEntropyCost():
    '''
    Class for the cross-entropy cost function, including the function evaluation, and the delta value from the output layer
    '''

    def __init__(self):
        self.flagwarned = 0

    @staticmethod
    def evaluate(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    def delta(self,z,a,y,activationf):
        # If the activation function of the output layer is chosen a sigmoid, the learning slowdown problem is solved: this is optimal
        if activationf.__class__.__name__ == 'SigmoidActivation':
            return a - y
        
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
        return -np.log(a[np.argmax(y)])
    
    def delta(self,z,a,y,activationf):
        # If the activation function of the output layer is chosen a sigmoid, the learning slowdown problem is solved: this is optimal
        if activationf.__class__.__name__ == 'SoftmaxActivation':
            return a - y
        
        else:
            if self.flagwarned == 0:
                print("######################################################\nWARNING: CHOICE OF OUTPUT LAYER ACTIVATION AND COST FUNCTION INEFFICIENT\n######################################################")
                self.flagwarned = 1
            return 


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
                 momentum = False):

        '''
        Initialize network with inputs:

            - neurons_layers (list) : number of neurons in each layer
            - activationf_hidden (class) : activation function of the neurons in the hidden layers
            - activationf_output (class) : activation function of the neurons in the output layer
            - costf (function) : cost function used to compute the cost
            - reg (string or None) : regularization type to use (None, L1, L2)
            - momentum (bool) : Indicates whether to use momentum-based gradient descent

        and attributes:

            - self.neurons_layers (list) : same value and function as neurons_layers
            - self.num_layers (int) : number of layers in the network (including the input and output layers)
            - self.initialize_weights initialises the weights and biases of the model as described in "self.initialize_weights()"
                * self.weights (list of np.arrays) : weights of each layer connections, where the position (i,j,k) represents the weight of the connection between neuron j from layer i to the neuron k from layer i-1
                * self.biases (list of np.arrays) : biases of each neuron in each layer, where the position (i,j) represents the bias of neuron j in layer i
            - self.activationf (list of classes) : activation function of the neurons in each layer
            - self.costf (function) : same value and function as cost
            - self.reg (string or None) : same value and function as reg
            - self.momentum (bool) : same value and function as bool. Initializing this parameter as True creates two new attributes for this model:
                * self.vw : 'Velocity' of the weights
                * self.vb : 'Velocity' of the biases
        '''
        self.neurons_layers = neurons_layers
        self.num_layers = len(neurons_layers)
        
        self.initialize_weights()

        if self.num_layers > 2:
            self.activationf = [activationf_hidden] * (self.num_layers-2) + [activationf_output]
        elif self.num_layers == 2:
            self.activationf = [activationf_output]
        else:
            raise Exception("Number of layers must be larger than 2")
        
        self.costf = costf
        self.reg = reg
        self.momentum = momentum
        if self.momentum:
            self.vw = [np.zeros(w.shape) for w in self.weights]
            self.vb = [np.zeros(b.shape) for b in self.biases]

        print(f"\n#### INITIALIZING NEURAL NETWORK WITH SETTINGS: ####\n   Number of layers: {self.num_layers}\n   Neurons per layer: {self.neurons_layers}\n   Activation function in hidden layers: {activationf_hidden.__class__.__name__}\n   Activation function in output layer: {activationf_output.__class__.__name__}\n   Cost function: {self.costf.__class__.__name__}")
        print(f"   No regularization") if not self.reg else print(f"   Regularization type: {self.reg}")
        print(f"   Ordinary gradient descent") if not self.momentum else print(f"   Momentum-based gradient descent\n")


    def initialize_weights(self):
        '''
        Initialize the weights and biases of the neural network in two different ways:

            - The weights are initialized using a normal distribution with mean zero and standard deviation 1 / sqrt(number of neurons connected to neuron)
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
        for n, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = (self.activationf[n]).evaluate(np.dot(w,a) + b)
        return a
    

    def stochastic_gradient_descent(self, trainingdata, batch_size, epochs, eta, lmbda = 0., mu = 0.,
                                    evaluationdata=None,
                                    monitor_training_cost = False,
                                    monitor_training_accuracy = False,
                                    monitor_evaluation_cost = False,
                                    monitor_evaluation_accuracy = False):
        '''
        Train the model using stochastic gradient descent

            - trainingdata (list of tuples (x, y)) : contains the (data, label) tuples to train the model
            - batch_size (int) : number of training examples per minibatch used
            - epochs (int) : number of epochs during which the model will be trained
            - eta (float) : learning rate
            - lmbda (float) : regularization parameter
            - mu (float) momentum co-efficient (only relevant if self.momentum = True)
        '''

        n = len(trainingdata)
        if evaluationdata: n_eval = len(evaluationdata)

        training_cost, training_accuracy, evaluation_cost, evaluation_accuracy= [], [], [], []

        for epoch in range(epochs):
            # At each epoch, shuffle the data, then divide it into subsets of size batch_size
            random.shuffle(trainingdata)
            mini_batches = [trainingdata[i:i+batch_size] for i in range(0,n,batch_size)]

            # For each batch update the parameters using backpropagation
            for minibatch in mini_batches:
                self.updateparams_minibatch(minibatch, eta, n, lmbda, mu)
            print(f"Epoch {epoch} complete")

            # Monitor the cost function on the training data
            if monitor_training_cost:
                epoch_cost = self.cost(trainingdata,lmbda)
                training_cost.append(epoch_cost)
                print(f"    Training cost: {epoch_cost}")

            # Monitor the accuracy on the training data
            if monitor_training_accuracy:
                epoch_accuracy = self.accuracy(trainingdata)
                training_accuracy.append(epoch_accuracy)
                print(f"    Training Accuracy: {epoch_accuracy}")

            # Monitor the cost function on the evaluation data
            if monitor_evaluation_cost and evaluationdata:
                epoch_cost = self.cost(evaluationdata,lmbda)
                evaluation_cost.append(epoch_cost)
                print(f"    evaluation cost: {epoch_cost}")
            
            # Monitor the accuracy on the evaluation data
            if monitor_evaluation_accuracy and evaluationdata:
                epoch_accuracy = self.accuracy(evaluationdata)
                evaluation_accuracy.append(epoch_accuracy)
                print(f"    evaluation Accuracy: {epoch_accuracy}")
        
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
        #Initialize the sum of the partial derivatives of the cost with respect to the weights and the biases as zero
        sum_pcost_pw = [np.zeros(w.shape) for w in self.weights]
        sum_pcost_pb = [np.zeros(b.shape) for b in self.biases]

        # Add up the aforementioned derivatives for every training example in the mini-batch
        for batch in minibatch:
            pcost_pw, pcost_pb = self.backprop(batch[0],batch[1])
            sum_pcost_pw = [sum_prev + new_term for sum_prev, new_term in zip(sum_pcost_pw, pcost_pw)]
            sum_pcost_pb = [sum_prev + new_term for sum_prev, new_term in zip(sum_pcost_pb, pcost_pb)]

        # Now the function is different depending on whether we use normal or momentum-based gradient descent
        if self.momentum:
            # L1 regularization
            if self.reg == 'L1':
                # Update velocities of the weights
                self.vw = [mu*v - eta*(np.sign(w)*lmbda/n + sum_learned_w) / len(minibatch) for v, w, sum_learned_w in zip(self.vw, self.weights, sum_pcost_pw)]

            # L2 regularization
            elif self.reg == 'L2':
                # Update velocities of the weights
                self.vw = [mu*v - eta*(w*lmbda/n + sum_learned_w) / len(minibatch) for v, w, sum_learned_w in zip(self.vw, self.weights, sum_pcost_pw)]

            # No regularization
            else:
                # Update velocities of the weights
                self.vw = [mu*v - eta * sum_learned_w / len(minibatch) for v, sum_learned_w in zip(self.vw, sum_pcost_pw)]

            # Update velocities of the biases
            self.vb = [mu*v - eta * sum_learned_b / len(minibatch) for v, sum_learned_b in zip(self.vb, sum_pcost_pb)]

            # Update weights and biases
            self.weights = [w + v for v, w in zip(self.vw, self.weights)]
            self.biases = [b + v for v, b in zip(self.vb, self.biases)]

        else:
            # L1 regularization
            if self.reg == 'L1':
                # Update the weights
                self.weights = [w - eta*(np.sign(w)*lmbda/n + sum_learned_w) / len(minibatch) for w, sum_learned_w in zip(self.weights, sum_pcost_pw)]

            # L2 regularization
            elif self.reg == 'L2':
                # Update weights
                self.weights = [(1 - eta*lmbda/n)*w - eta * sum_learned_w / len(minibatch) for w, sum_learned_w in zip(self.weights, sum_pcost_pw)]

            # No regularization
            else:
                # Update weights
                self.weights = [w - eta * sum_learned_w / len(minibatch) for w, sum_learned_w in zip(self.weights, sum_pcost_pw)]

            # Update biases
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
            a_layers.append((self.activationf[n]).evaluate(z_layers[n]))

        # Now we get the delta of the final layer
        delta = (self.costf).delta(z_layers[-1], a_layers[-1], y, self.activationf[-1])
        pcost_pb[-1] = delta
        pcost_pw[-1] = np.dot(delta,a_layers[-2].T)

        # Propagate backwards: loop from the second to last layer, backwards
        for l in range(2,self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * (self.activationf[-l]).derivative(z_layers[-l])
            pcost_pb[-l] = delta
            pcost_pw[-l] = np.dot(delta, a_layers[-l-1].T)

        # Lastly return the resulting matrices
        return pcost_pw, pcost_pb
    
    def accuracy(self, data):
        '''
        Returns the accuracy (%) of the model in predicting the labels of the current dataset

            - data (list of tuples) : tuples (rawdata, label) on which the model will be evaluated
        
        where rawdata is the raw data, and label is a list representing which neuron represents the correct result (so all zeros except for one)
        '''
        return 100. * np.sum([np.argmax(self.feedforward(rawdata)) == np.argmax(label) for rawdata, label in data])/len(data)

    def cost(self, data, lmbda):
        '''
        Returns the total cost of the data given this model
        The arguments of the method are analogous to those from self.accuracy
        '''
        cost = 0.
        for rawdata, label in data:
            cost += (self.costf).evaluate(np.argmax(self.feedforward(rawdata)),np.argmax(label))
        
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
class Network():

    def __init__(self, layered_neurons, costf):

        '''
        Initialize network with:

            - self.num_layers (int): number of layers in the network (including the input and output layers)
            - self.layered_neurons (list): number of neurons in each layer
            - self.costf (function) : cost function used to compute the cost

        '''
        self.num_layers = len(layered_neurons)
        self.layered_neurons = layered_neurons
        self.initialize_weights()
        self.costf = costf

    

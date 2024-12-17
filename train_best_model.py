import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import json

from neuralnet import *
sys.path.append(".")
from load_mnist import MNISTLoader


def json_save(filename,data):
    '''
    Saves data to json file
    '''
    with open(filename,'w') as outfile:
        json.dump(data,outfile)
        
    return


def json_load(filename, print_terminal=True):
    '''
    Loads data from json file
    '''
    with open(filename,'r') as infile:
        data = json.load(infile)

    if print_terminal:
        print(f"File {filename} successfully loaded")

    return data


def initialize_data(filename = None):
    '''
    Initializes the data dictionary of dictionaries. If filename is not None, instead of initializing the empty dataset, it will load an existing one at 'filename'
    '''
    if filename == None or not os.path.isfile(filename):
        data = {}

        data['traincost'], data['trainacc'] = {}, {}
        data['evaluationcost'], data['evaluationacc'] = {}, {} 
        data['testcost'], data['testacc'] = {}, {}
        data['time'] = {}

    else:
        data = json_load(filename)

    return data


def update_data(data, train_cost, train_accuracy, eval_cost, eval_accuracy, test_cost, test_accuracy, time, label):
    '''
    Updates the data (train and evaluation data cost and accuracy as a function of epoch)
    '''

    data['traincost'][label] = train_cost
    data['trainacc'][label] = train_accuracy

    data['evaluationcost'][label] = eval_cost
    data['evaluationacc'][label] = eval_accuracy

    data['testcost'][label] = test_cost
    data['testacc'][label] = test_accuracy

    data['time'][label] = time


def load_bestparams(filename):
    '''
    Loads the best parameters found from the file created by basic_hyperparameter_search.py
    '''
    # Load the data    
    data = json_load(filename)
    
    best_params = {}
    for key in data:
        # Find the parameter index that yields the higher accuracy
        max_acc_value_index = np.argmax([max(acc) for acc in data[key]['acc']])

        # Having the index, we introduce the parameter into best_params
        if key == 'eta':
            # The case for eta is special, as we need both eta and eta_var. Furthermore, since eta_var represents functions and they could not be saved into json format, we need to load the corresponding function manually
            etas = [0.1, 0.05, 0.01, lambda x: 0.1 if x >= 0.1 else 0.5 * x, lambda x: 0.1 if x >= 0.1 else 0.2 * x, lambda x: 0.1 if x<=60 else 0.1*np.exp(1-x/60)]
            etas_var = [None,None,None,'evacc','evacc','epoch']
            best_params[key] = etas[max_acc_value_index]
            best_params['eta_var'] = etas_var[max_acc_value_index]

        else:
            best_params[key] = data[key]['values'][max_acc_value_index]

    return best_params


def train_network(trainingdata, evaluationdata, testingdata, network_params, data, save_path, outfile_name):
    '''
    Initializes and trains a neural network with the data and parameters specified in the input
        - trainingdata, evaluationdata, testingdata: training, evaluation and testing data, as obtained from as returned by MNISTLoader.process_all()
        - network_params: dictionary of network parameters
        - data: data object to update with the results
        - save_path: path where to save the models and the data
        - outfile_name: filename of the data json file
    '''
    net = Network(network_params['net'], 
                activationf_hidden = network_params['activationf_hidden'], 
                activationf_output = network_params['activationf_output'], 
                costf = network_params['costf'],
                reg = network_params['reg'],
                momentum = network_params['momentum'])

    t_train_0 = time.time()             # Variable to measure training time
    monitor = True

    # Begin training using the best hyperparameters
    t_cost, t_accuracy, ev_cost, ev_accuracy = net.stochastic_gradient_descent(
        trainingdata, evaluationdata,
        batch_size = network_params['batch'], epochs = network_params['epochs'],
        eta= network_params['eta'], eta_var = network_params['eta_var'], lmbda = network_params['lmbda'], mu = network_params['mu'],
        dynamic_stop = network_params['dynamic_stop'],
        monitor_training_cost = monitor,
        monitor_training_accuracy = monitor,
        monitor_evaluation_cost = monitor,
        monitor_evaluation_accuracy = monitor)
    t_to_train = time.time() - t_train_0

    # Having trained the model, we save the model, and all the relevant data
    net.savemodel(os.path.join(save_path,f'{network_params["label"]}.json'))
    update_data(data, 
                t_cost, t_accuracy, ev_cost, ev_accuracy, 
                net.cost(testingdata, network_params['lmbda']), net.accuracy(testingdata),
                t_to_train, label=network_params['label'])
    json_save(os.path.join(save_path,outfile_name),data)


def plot_data(data, filename):
    '''
    Plots the data obtained using the basic_search function. Saves to filename one figure containing a 2x1 subplot with the cost and accuracy of running the models on the evaluation data as a function of epoch, and shows the best accuracy obtained, the model and the epoch
        - data: data to plot, obtained from "basicsearch"
        - filename: file where to save the resulting plot
    '''

    labels = [r'$\sigma \rightarrow \sigma \rightarrow \mathcal{L}_{cross-entropy}$',r'$\sigma \rightarrow softmax \rightarrow \mathcal{L}_{log-likely}$',r'$ReLU \rightarrow softmax \rightarrow \mathcal{L}_{log-likely}$']
    labels = {key:label for key, label in zip(data['evaluationacc'], labels)}

    fig, axs = plt.subplots(2,1,figsize=(6,8))
    axs[1].set_xlabel('Epoch')
    axs[0].set_ylabel('Cost')
    axs[1].set_ylabel('Accuracy')

    for key, label in labels.items():

        axs[0].plot(data['evaluationcost'][key], label = label)
        axs[1].plot(data['evaluationacc'][key])

    best_acc_model = max(data['evaluationacc'], key = data['evaluationacc'].get)
    best_acc_index = np.argmax(data['evaluationacc'][best_acc_model])
    axs[1].text(20,88,f'Highest accuracy: {data["evaluationacc"][best_acc_model][best_acc_index]:.2f} % at epoch {best_acc_index}\n Model: {labels[best_acc_model]}')

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(filename,dpi=200)
    plt.close()

    print(f"Models trained in: {[time/3600 for time in data['time'].values()]} hours")
    print(f"Cost of running on testing data: {[cost for cost in data['testcost'].values()]}")
    print(f"Accuracy of running on testing data: {[acc for acc in data['testacc'].values()]}")


def main():

    # Load the MNIST data
    trainingdata_path = r"..\data_MNIST\MNIST\raw\train-images-idx3-ubyte.gz"
    traininglabels_path = r"..\data_MNIST\MNIST\raw\train-labels-idx1-ubyte.gz"
    testingdata_path = r"..\data_MNIST\MNIST\raw\t10k-images-idx3-ubyte.gz"
    testinglabels_path = r"..\data_MNIST\MNIST\raw\t10k-labels-idx1-ubyte.gz"

    loader = MNISTLoader(trainingdata_path, traininglabels_path, testingdata_path, testinglabels_path)
    training, testing = loader.process_all()

    # Load the best hyperparameters found in basic_hyperparameter_serach.py
    best_params = load_bestparams(os.path.join('plots','hyperparams_study','data_hyperparams.json'))

    # Dictionary that will contain the data for each model, mainly the evolution
    path_to_data = os.path.join('plots','best_model')
    data = initialize_data(filename = os.path.join(path_to_data,'bestmodels_data.json'))

    # Initialize the configurations for the networks we will train
    configurations = [
        {
            'net' : best_params['nets'],
            'activationf_hidden' : SigmoidActivation(),
            'activationf_output' : SigmoidActivation(),
            'costf' : CrossEntropyCost(),
            'reg' : best_params['regs'],
            'momentum': True,
            'batch' : best_params['batches'],
            'epochs': 400,
            'eta' : best_params['etas'],
            'eta_var': best_params['etas_var'],
            'lmbda' : best_params['lmbdas'],
            'mu' : best_params['mus'],
            'dynamic_stop': (15,1e-2),
            'label': 'sigmoid-sigmoid-crossentropy'

        },
        {
            'net' : best_params['nets'],
            'activationf_hidden' : SigmoidActivation(),
            'activationf_output' : SoftmaxActivation(),
            'costf' : LoglikelyCost(),
            'reg' : best_params['regs'],
            'momentum': True,
            'batch' : best_params['batches'],
            'epochs': 400,
            'eta' : best_params['etas'],
            'eta_var': best_params['etas_var'],
            'lmbda' : best_params['lmbdas'],
            'mu' : best_params['mus'],
            'dynamic_stop': (15,1e-2),
            'label': 'sigmoid-softmax-loglikely'

        },
        {
            'net' : best_params['nets'],
            'activationf_hidden' : ReluActivation(),
            'activationf_output' : SoftmaxActivation(),
            'costf' : LoglikelyCost(),
            'reg' : best_params['regs'],
            'momentum': True,
            'batch' : best_params['batches'],
            'epochs': 400,
            'eta' : lambda x: 0.1 if x<=30 else 0.1*np.exp(1-x/30),
            'eta_var': best_params['etas_var'],
            'lmbda' : best_params['lmbdas'],
            'mu' : best_params['mus'],
            'dynamic_stop': (15,1e-2),
            'label': 'relu-softmax-loglikely'

        }
    ]

    # Train these models
    for config in configurations:
        train_network(training[:50000], training[50000:60000], testing, config, data, path_to_data, 'bestmodels_data.json')

    # Having trained the models and saved the data, we plot them
    plot_data(data,os.path.join(path_to_data,'results.png'))
    

if __name__ == '__main__':
    main()


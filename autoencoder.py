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


def train_network_with_pretrain(trainingdata, evaluationdata, testingdata, network_params, outfile):
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
                momentum = network_params['momentum'],
                print_terminal=network_params['print_terminal'])
    
    t_pretrain = time.time()
    monitor = True

    # Pre-train the model
    net.pretrain_GreedyLayerAutoencoder(
        trainingdata, evaluationdata,
        pretrain_activationf_hidden = network_params['pretrain_activationf_hidden'], pretrain_activationf_output = network_params['pretrain_activationf_output'],
        pretrain_costf = network_params['pretrain_costf'],
        batch_size = network_params['batch'], epochs = network_params['epochs'],
        eta= network_params['eta'], eta_var = network_params['eta_var'], lmbda = network_params['lmbda'], mu = network_params['mu'],
        dynamic_stop = network_params['dynamic_stop'],
        monitor_training_cost = monitor,
        monitor_training_accuracy = monitor,
        monitor_evaluation_cost = monitor,
        monitor_evaluation_accuracy = monitor,
        print_terminal=network_params['print_terminal']
        )
    
    t_pretrain = time.time() - t_pretrain

    t_train = time.time()             # Variable to measure training time

    # Begin training
    t_cost, t_accuracy, ev_cost, ev_accuracy = net.stochastic_gradient_descent(
        trainingdata, evaluationdata,
        batch_size = network_params['batch'], epochs = network_params['epochs'],
        eta= network_params['eta'], eta_var = network_params['eta_var'], lmbda = network_params['lmbda'], mu = network_params['mu'],
        dynamic_stop = network_params['dynamic_stop'],
        monitor_training_cost = monitor,
        monitor_training_accuracy = monitor,
        monitor_evaluation_cost = monitor,
        monitor_evaluation_accuracy = monitor
        )
    
    t_train = time.time() - t_train

    # Return the training time and the accuracy on the testing data
    json_save(outfile,{'pretrain_time':t_pretrain,
                       'train_time':t_train,
                       'trainingcost':t_cost,
                       'trainingacc':t_accuracy,
                       'evaluationcost':ev_cost,
                       'evaluationaccuracy':ev_accuracy,
                       'testingcost':net.cost(testingdata,network_params['lmbda']),
                       'testingacc':net.accuracy(testingdata)})


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
    print("Figure saved at {filename}")
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

    # Initialize the configurations for the networks we will train
    configuration = {
        'net' : [784,480,10],
        'activationf_hidden' : ReluActivation(),
        'activationf_output' : SoftmaxActivation(),
        'costf' : LoglikelyCost(),
        'pretrain_activationf_hidden': SigmoidActivation(),
        'pretrain_activationf_output': SigmoidActivation(),
        'pretrain_costf': CrossEntropyCost(),
        'reg' : 'L2',
        'momentum': True,
        'batch' : 160,
        'epochs': 400,
        'eta' : lambda x: 0.1 if x<=30 else 0.1*np.exp(1-x/30),
        'eta_var': 'epoch',
        'lmbda' : 0.1,
        'mu' : 0.3,
        'dynamic_stop': (20,1e-2),
        'print_terminal':True
    }

    # Train these models
    train_network_with_pretrain(training[:50000], training[50000:60000], testing, configuration, 'pretrain_data.json')
    # Having trained the models and saved the data, we plot them
    #plot_data(data,os.path.join(path_to_data,'results.png'))
    

if __name__ == '__main__':
    main()

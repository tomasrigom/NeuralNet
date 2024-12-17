import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import json

from neuralnet import *
sys.path.append(".")
from load_mnist import MNISTLoader


def json_save(data, filename):
    '''
    Saves the dictionary "data" to the file "filename" in json format
    '''
    with open(filename,'w') as outfile:
        json.dump(data,outfile)


def json_load(filename):
    '''
    Loads the dictionary from the file "filename" in json format
    '''
    with open(filename,'r') as infile:
        data = json.load(infile)

    return data


def train_network(trainingdata, evaluationdata, network_params):
    '''
    Initializes and trains a neural network with the data and parameters specified in the input
        - trainingdata, evaluationdata: training and evaluation data, as obtained from as returned by MNISTLoader.process_all()
        - network_params: dictionary of network parameters
    '''
    # Initialize network
    net = Network(network_params['net'], 
                activationf_hidden = network_params['activationf_hidden'], 
                activationf_output = network_params['activationf_output'], 
                costf = network_params['costf'],
                reg = network_params['reg'],
                momentum = network_params['momentum'])

    # Begin training using the chosen hyperparameters
    monitor = True
    t_cost, t_accuracy, ev_cost, ev_accuracy = net.stochastic_gradient_descent(
        trainingdata, evaluationdata,
        batch_size = network_params['batch'], epochs = network_params['epochs'],
        eta= network_params['eta'], eta_var = network_params['eta_var'], lmbda = network_params['lmbda'], mu = network_params['mu'],
        dynamic_stop = network_params['dynamic_stop'],
        monitor_training_cost = monitor,
        monitor_training_accuracy = monitor,
        monitor_evaluation_cost = monitor,
        monitor_evaluation_accuracy = monitor)

    return t_cost, t_accuracy, ev_cost, ev_accuracy


def basicsearch(training, evaluation, trials_values_dict, default_params = False, datafile = False, data_load = False):
    '''
    Performs a basic study of the performance of a neural network, varying one parameter at a time. Returns a dictionary with the cost and accuracy of the model during training on the evaluation data
        - training, evaluation: training and evaluation data, as returned by MNISTLoader.process_all()
        - trials_values_dict: dictionary in which the parameters we will try are specified
        - default_params: choice of default parameters. If set to False, the in-built default hyperparameters from this function are used
        - datafile: file where to save the data (and its checkpoints) in .json format
        - data_load: If set to True, the data from datafile will be loaded before making any modifications
    '''

    # Dictionary used for saving the data
    if data_load and datafile:
        data = json_load(datafile)
    else:
        data = {}

    # Default configuration
    if not default_params:
        config = {
                'net' : [784,30,10],
                'activationf_hidden' : SigmoidActivation(),
                'activationf_output' : SigmoidActivation(),
                'costf' : CrossEntropyCost(),
                'reg' : 'L2',
                'momentum': True,
                'batch' : 100,
                'epochs': 400,
                'eta' : 0.1,
                'eta_var': None,
                'lmbda' : 0.1,
                'mu' : 0.1,
                'dynamic_stop': (15,1e-2),
            }
        
    else:
        config = default_params
    
    # We loop over the configurations specified by the user to test the parameters. Only one parameter can be changed at a time
    for trials_key, trials_values in trials_values_dict.items():
        # Lists for saving the results (cost and accuracy) of the different trials
        ev_cost_trials = []
        ev_acc_trials = []

        # Copy the default configuration to modify it
        trial_config = {k:v for k, v in config.items()}
        
        # The case of eta is special, as we also need to specify whether it varies or not
        if trials_key=='eta':
            # Loop over the trial values
            for trial_eta, trial_eta_var in zip(trials_values,trials_values_dict['eta_var']):
                # Modify the configuration for this trial from the default one
                trial_config[trials_key] = trial_eta
                trial_config['eta_var'] = trial_eta_var

                # Train the model with this configuration
                t_cost, t_accuracy, ev_cost, ev_accuracy = train_network(training,evaluation,trial_config)

                # Add the data to the list of lists
                ev_cost_trials.append(ev_cost)
                ev_acc_trials.append(ev_accuracy)

            # Having iterated over all the values for this particular parameter, we update the data and save it (in case any error happens along the way)
            data[trials_key] = {'values':trials_values, 'cost':ev_cost_trials, 'acc':ev_acc_trials}
            if datafile:
                json_save(data, datafile)
        
        # Since we used both eta and eta_var together, we skip the latter
        elif trials_key=='eta_var':
            continue
        
        # The rest of the cases only require a single hyperparameter
        else:
            for value in trials_values:
                trial_config[trials_key] = value

                t_cost, t_accuracy, ev_cost, ev_accuracy = train_network(training,evaluation,trial_config)
            
                ev_cost_trials.append(ev_cost)
                ev_acc_trials.append(ev_accuracy)

            data[trials_key] = {'values':trials_values, 'cost':ev_cost_trials, 'acc':ev_acc_trials}
            if datafile:
                json_save(data, datafile)
            
            return data


def plot_data_basicsearch(data, labels, textpos, savepath):
    '''
    Plots the data obtained using the basic_search function. Saves to savepath one figure containing a 2x1 subplot with the cost and accuracy of running the model on the evaluation data as a function of epoch, and shows the best accuracy obtained, the parameter value and the epoch
        - data: data to plot, obtained from "basicsearch"
        - labels: dictionary of labels for the legend of the plots
        - textpos: position of the text in the figures
        - savepath: where to save the plots
    '''
    for key, label in zip(data, labels):
        fig, axs = plt.subplots(2,1,figsize=(6,8))

        axs[1].set_xlabel('Epoch')
        axs[0].set_ylabel('Cost')
        axs[1].set_ylabel('Accuracy')

        for i, value in enumerate(data[key]['values']):
            axs[0].plot(data[key]['cost'][i],label = label+f'{value}')

            # This has been commented out as i found it more useful to look at the accuracy rather than the cost
            # min_cost_value_index = np.argmin([min(cost) for cost in data[key]['cost']])
            # min_cost_epoch = np.argmin(data[key]['cost'][min_cost_value_index])
            # axs[0].text(*textpos_data[key][0],f"Min cost: {data[key]['cost'][min_cost_value_index][min_cost_epoch]:.2f}\nFor {key}: {data[key]['values'][min_cost_value_index]}\nEpoch:{min_cost_epoch}")

            axs[1].plot(data[key]['acc'][i])

            max_acc_value_index = np.argmax([max(acc) for acc in data[key]['acc']])
            max_acc_epoch = np.argmax(data[key]['acc'][max_acc_value_index])
            axs[1].text(*textpos[key][1],f"Max accuracy: {data[key]['acc'][max_acc_value_index][max_acc_epoch]:.2f}\nFor {key}: {data[key]['values'][max_acc_value_index]}\nEpoch:{max_acc_epoch}")
            
        axs[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savepath,f'{key}.png'),dpi=200)
        plt.close()


def main():
    # Load MNIST data
    trainingdata_path = r"..\data_MNIST\MNIST\raw\train-images-idx3-ubyte.gz"
    traininglabels_path = r"..\data_MNIST\MNIST\raw\train-labels-idx1-ubyte.gz"
    testingdata_path = r"..\data_MNIST\MNIST\raw\t10k-images-idx3-ubyte.gz"
    testinglabels_path = r"..\data_MNIST\MNIST\raw\t10k-labels-idx1-ubyte.gz"

    loader = MNISTLoader(trainingdata_path, traininglabels_path, testingdata_path, testinglabels_path)
    training, testing = loader.process_all()

    # Parameters used for initial optimization study
    trials = {
        'net': [[784,30,10],[784,60,10],[784,120,10], [784,480,10], [784,60,30,10]],
        'reg': ['L1','L2'],
        'batch': [40,80,160,400],
        'eta': [0.1, 0.05, 0.01, lambda x: 0.1 if x >= 0.1 else 0.5 * x, lambda x: 0.1 if x >= 0.1 else 0.2 * x, lambda x: 0.1 if x<=60 else 0.1*np.exp(1-x/60)],
        'eta_var': [None,None,None,'evacc','evacc','epoch'],
        'lmbda': [0.3, 0.1, 0.05, 0.01],
        'mu': [0.3,0.1,0.01]
    }

    # Obtain data for the settings specified: perform study
    path_to_data = os.path.join('plots','hyperparams_study','data_hyperparams.json')
    data = basicsearch(training[:50000], training[50000:60000],
                trials, datafile = path_to_data, data_load=False)

    # Now we plot the data: cost and accuracy against epoch
    labels_data = [r'$\eta=$',r'$\lambda=$',r'$n_{batch}=$',r'$\mu=$','reg'+r'$=$','net'+r'$=$']
    textpos_data = {'etas':[[80,1],[200,70]],'lmbdas':[[40,0.4],[50,88]],'batches':[[60,0.7],[100,75]],'mus':[[35,0.45],[60,88]],'regs':[[40,0.45],[50,88]],'nets':[[40,0.6],[70,70]]}

    # Produce plots
    plot_data_basicsearch(data, labels_data, textpos_data, os.path.join('plots','hyperparams_study'))

if __name__ == '__main__':
    main()

